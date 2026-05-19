import sys, logging, pyautogui, time, os
import numpy as np, pandas as pd
from watchdog.observers import Observer
import watchdog.events
import datafile_manager, ble_comms

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "a")
        self.need_retry = False  # Important for watchdog to restart exp if continue button is not found

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ------------------------------------------------------------
# FILE HANDLER
# ------------------------------------------------------------
class FileHandler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self, filename, controller):
        super().__init__(patterns=[filename])
        self.controller = controller
        self.processing = False

    def on_modified(self, event):
        if self.processing:
            # print("Already processing, skipping duplicate event...")
            return
        
        if not os.path.exists(event.src_path):
            # print("File not ready yet, skipping...")
            return
        
        print("Datafile was modified! - % s" % event.src_path)
        self.processing = True
        time.sleep(3)
        self.controller.process_datafile()
        self.processing = False


# ------------------------------------------------------------
# CONTROLLER
# ------------------------------------------------------------
class ExperimentController:
    def __init__(self, file, file_path):

        self.filename = file
        self.datafile_path = file_path
        self.datafile_name = os.path.join(file_path, f"{file}.csv")  # Use os.path.join

        # Logging
        # Start logging to file without removing terminal output
        sys.stdout = TeeLogger(f"{self.filename}.txt")
        sys.stderr = sys.stdout

        # BLE Communication Init
        ble_comms.connect_device("COM7", 115200, 0.1)
        self.log_file = open(f"{self.filename}_b.out", "a")

        # Blanks
        self.od_blank = np.ones(32) * 0.15
        # self.od_blank = [0.15, 0.17, 0.14, 0.17, 0.16, 0.12, 0.11, 0.1,
        #                  0.11, 0.12, 0.12, 0.16, 0.13, 0.16, 0.12, 0.18]
        self.fl_blank = np.ones(32) * 5

        # Wells
        self.ctrl_wells = [
            'A11', 'B11', 'C11', 'D11', 'E11', 'F11', 'G11', 'H11',
            'H9',  'G9',  'F9',  'E9',  'D9',  'C9',  'B9',  'A9'
        ]
        self.neg_well = 'H11'
        self.neg_index = self.ctrl_wells.index(self.neg_well)

        self.g_ctrl_wells = ['A9', 'C9', 'B11', 'D11']
        self.r_ctrl_wells = ['B9', 'D9', 'E9', 'F9', 'A11', 'C11', 'E11', 'F11']
        self.dark_wells = ['G9', 'H9', 'H11', 'G11']

        self.stpt1_P_wells = []
        self.stpt2_P_wells = []
        self.stpt1_PI_wells = []
        self.stpt2_PI_wells = []
        self.stpt1_PID_wells = []
        self.stpt2_PID_wells = []

        self.p_wells = self.stpt1_P_wells + self.stpt2_P_wells
        self.pi_wells = self.stpt1_PI_wells + self.stpt2_PI_wells
        self.pid_wells = self.stpt1_PID_wells + self.stpt2_PID_wells

        self.measurement_interval = 600
        self.default_on_time = (self.measurement_interval - 120) / 2
        self.max_on_time = self.measurement_interval - 120

        self.p_gain = 0.064
        self.pi_gain = 0.013
        self.tau_I = 1500 * 60      # <- Integral time constant in seconds
        self.K_pid = 0.032          # overall gain
        self.tau_I_pid = 1500 * 60  # integral time constant in seconds
        self.tau_D_pid = 120 * 60   # derivative time constant in seconds

        self.errors = {w: [] for w in self.ctrl_wells}
        self.pi_integrals = {w: 0.0 for w in self.pi_wells + self.pid_wells}
        self.ctrl_times_stamp = []
        self.ctrl_times = [self.default_on_time] * 32

        self.error_df = pd.DataFrame(columns=["time"] + self.pi_wells + self.pid_wells)

        self.started = False
        self.start_time = 0
        self.overall_start = 0

        # set up file system watcher
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # ------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------
    def run_experiment(self):
        print("Starting in 2 seconds...")
        time.sleep(2)

        self._click_image("1.run", retries=3)
    
        # Try to click continue, retry run if continue not found
        if self._click_image("2.continue", retries=3, sleep_time=6):
            # Success!
            self.need_retry = False
        else:
            print("Continue button not found, will retry after export...")
            self.need_retry = True
            time.sleep(20)
            self._click_image("3.export", retries=3)
            return
        
        # Try to click OK, retry run if continue not found
        if self._click_image("3.loadplate", retries=3):
            # Success!
            self.need_retry = False
        else:
            print("OK button not found, will retry after export...")
            self.need_retry = True
            time.sleep(20)
            self._click_image("3.export", retries=3)
            return

        self.start_time = time.time()
        if not self.started:
            self.overall_start = time.time()
        self.started = True

    def _click_image(self, name, retries=3, wait=5, sleep_time=2):
        for _ in range(retries):
            try:
                loc = pyautogui.center(pyautogui.locateOnScreen(f"search_targets/{name}.PNG"))
                pyautogui.click(loc)
                time.sleep(sleep_time)
                return True
            except:
                time.sleep(wait)
        print("Could not find", name)
        return False

    # ------------------------------------------------------------
    # TIMING
    # ------------------------------------------------------------
    def handle_timing(self):
        if not self.started:
            # print("Experiment not started yet.")
            return

        curr_time = time.time() - self.start_time
        corrected_time = curr_time % self.measurement_interval
        to_send = ""

        for well in self.ctrl_wells:
            if corrected_time > 60 and corrected_time < (self.measurement_interval - 60):
                if well in self.g_ctrl_wells:
                    to_send += "g" 
                elif well in self.r_ctrl_wells:
                    to_send += "r" 
                elif well in self.dark_wells:
                    to_send += "o" 
                else:
                    to_send += "o"
            else:
                to_send += "o"

        to_send += "\n"
        ble_comms.write_data(to_send, self.log_file, curr_time + self.start_time - self.overall_start)

    # ------------------------------------------------------------
    # PROCESS DATAFILE
    # ------------------------------------------------------------
    def process_datafile(self):
        # Add file existence check
        # print(f"Attempting to read: {self.datafile_name}")
        # print(f"File exists: {os.path.exists(self.datafile_name)}")
        
        if not os.path.exists(self.datafile_name):
            # print("File doesn't exist yet, skipping processing")
            if self.need_retry:
                # print("Will retry experiment...")
                time.sleep(2)
                self.run_experiment()
            return
        
        try:
            datafile_manager.read_and_save(self.datafile_name)
        except BaseException as e:
            print("Error reading csv export:", e)
            if self.need_retry:
                print("Error occurred during retry, will try again...")
                time.sleep(2)
                self.run_experiment()
            return

        # get latest rows of each table
        fl_latest = datafile_manager.get_fl_latest()
        od_latest = datafile_manager.get_od_latest()

        newest_idx = list(fl_latest[self.ctrl_wells[self.neg_index]].keys())[-1]
        neg_fl = int(fl_latest[self.neg_well][newest_idx]) - self.fl_blank[self.neg_index]
        neg_od = float(od_latest[self.neg_well][newest_idx]) - self.od_blank[self.neg_index]

        curr_time = (time.time() - self.overall_start) / 1 # Keep it in seconds only
        self.ctrl_times_stamp.append(curr_time)

        for i, well in enumerate(self.ctrl_wells):
            od = od_latest[well][newest_idx]
            od = 1 if str(od).upper() == "OVRFLW" else min(float(od), 32) - self.od_blank[i] # Sometimes OD can be OVRFLW if there is something in the well (like a stuck well mold)
            fl = fl_latest[well][newest_idx]
            fl = 100000 if str(fl).upper() == "OVRFLW" else int(fl) - self.fl_blank[i]

            # print("processing well", self.ctrl_wells[i], "with fl", str(fl), "and od", str(od))
            fl_by_od = (fl / od) - (neg_fl / neg_od)

            # Setpoints are same for P, PI and PID control
            st_pt_1 = 11500
            st_pt_2 = 18500
            ctrl_pt = st_pt_1
            
            error = ctrl_pt - fl_by_od
            self.errors[well].append(error) # Save error for all wells

            if well in self.pid_wells:
                # PID Control Wells 
                if len(self.ctrl_times_stamp) >= 2:
                    self.pi_integrals[well] = self.compute_integral(well)
                    deriv = self.compute_derivative(well)
                    # Apply your requested formula
                    on = self.K_pid * (error + self.tau_D_pid*deriv + (1/self.tau_I_pid)*self.pi_integrals[well])
                else:
                    on = self.K_pid * error # Not enough data points yet

            elif well in self.pi_wells:
                # PI Control Wells
                if len(self.ctrl_times_stamp) >= 2:
                    self.pi_integrals[well] = self.compute_integral(well)
                    on = self.pi_gain * (error + (1/self.tau_I)*self.pi_integrals[well])
                else:
                    on = self.pi_gain * error

            else:
                # P Control Wells
                on = self.p_gain * error

            on_time = max(0, min(self.max_on_time, on))
            # print("calculated raw on_time of:", on_time, "seconds")
            self.ctrl_times[i] = on_time
            # print()
        # print("New duration setpoints calculated: ", self.ctrl_times)

        # Save errors to CSV
        row = {"time": curr_time}
        # Combine PI and PID wells
        for well in self.pi_wells + self.pid_wells:
            if len(self.errors[well]) > 0:
                row[well] = self.errors[well][-1]
            else:
                row[well] = ""
        new_row = pd.DataFrame([row])

        nonempty = [c for c in new_row.columns if pd.notna(new_row[c]).any()]
        new_row = new_row[nonempty]

        self.error_df = pd.concat([self.error_df, new_row], ignore_index=True)
        self.error_df.to_csv('Datafile/errors.csv', index=False)

        # delete the datafile, a new one will be created in the next iteration
        
        try:
            datafile_manager.remove()
        except BaseException as e:
            print("Error deleting csv export...", e)
        
        print("Restarting experiment after processing datafile...")
        time.sleep(2)  # Brief pause before retry
        self.run_experiment()

        # try:
        #     self.run_experiment()
        # except BaseException as e:
        #     print("Error running a new experiment...", e)
        # either remove this one or the one in file handler

    # ------------------------------------------------------------
    # INTEGRAL / DERIVATIVE
    # ------------------------------------------------------------
    def compute_integral(self, well):
        """
        Computes full trapezoidal integral of error history from time = 0.
        """
        if len(self.ctrl_times_stamp) < 2 or len(self.errors[well]) < 2: return 0.0 # Not enough points for integral computation
        integral = 0.0
        errs = self.errors[well]
        for i in range(1, len(self.ctrl_times_stamp)):
            dt = self.ctrl_times_stamp[i] - self.ctrl_times_stamp[i-1]
            integral += 0.5 * (errs[i] + errs[i-1]) * dt
        return integral

    def compute_derivative(self, well):
        """
        Computes derivative of the error (finite difference).
        """
        if len(self.ctrl_times_stamp) < 2 or len(self.errors[well]) < 2: return 0.0 # Not enough points for derivative computation
        dt = self.ctrl_times_stamp[-1] - self.ctrl_times_stamp[-2]
        if dt == 0: return 0.0
        return (self.errors[well][-1] - self.errors[well][-2]) / dt

    # ------------------------------------------------------------
    # RUN LOOP
    # ------------------------------------------------------------
    def run(self):
        try:
            self.run_experiment()
            while True:
                self.handle_timing()
                msg = ble_comms.read_data()
                if msg != "":
                    print(msg)
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Shutting down experiment.")
            self.log_file.close()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    file = "test_031326"
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."
    file_path = os.path.join(base_path, "Datafile")

    controller = ExperimentController(file, file_path)

    datafile_event_handler = FileHandler(f"{file}.csv", controller)
    datafile_observer = Observer()
    datafile_observer.schedule(datafile_event_handler, file_path, recursive=True)
    datafile_observer.start()

    controller.run()
