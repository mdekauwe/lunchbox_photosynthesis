import time
from collections import deque
import numpy as np
import statsmodels.api as sm
from scipy.signal import savgol_filter, butter, filtfilt, medfilt

from xensiv_pas_co2_sensor import CO2Sensor
from serial_port_finder import find_usb_port

class LunchboxLogger:
    def __init__(self, port, baud, lunchbox_volume, temp_c, leaf_area_cm2,
                 window_size, measure_interval=10, timeout=1.0, smoothing=True,
                 rolling_regression=False, area_basis=True,
                 soil_resp_correction=0.0,):

        self.temp_k = temp_c + 273.15
        self.pressure = 101325.0  # Pa
        self.leaf_area_m2 = leaf_area_cm2 / 10000.0
        self.lunchbox_volume = lunchbox_volume
        self.window_size = window_size
        self.measure_interval = measure_interval
        self.area_basis = area_basis
        self.soil_resp_correction = soil_resp_correction
        self.smoothing = smoothing
        self.rolling_regression = rolling_regression

        # Data buffers
        self.co2_window = deque(maxlen=window_size)
        self.time_window = deque(maxlen=window_size)
        self.last_co2 = None

        # Setup sensor
        self.sensor = CO2Sensor(port, baud, timeout)
        try:
            self.sensor.reset_sensor()
            self.sensor.set_pressure_reference(101325) # default was 900 hpa
            time.sleep(2)
            self.sensor.arm_sensor(rate_seconds=self.measure_interval)
        except Exception as e:
            print(f"Failed to arm sensor: {e}")
            self.sensor.close()
            raise

        self.last_measure_time = 0
        self.start_time = time.time()

    def calc_anet(self, delta_ppm_s):
        # Net assimilation rate (An_leaf, umol leaf-1 s-1) calculated using the
        # ideal gas law to solve for "n" amount of substance, moles of gas
        # i.e, converts ppm s-1 into umol s-1
        #
        #            delta_CO2 × p × V
        # An_leaf = -------------------
        #                  R × T
        #
        # where:
        #   delta_CO2 = rate of CO2 change (ppm s-1)
        #   p         = pressure (Pa)
        #   V         = lunchbox_volume (m3)
        #   R         = universal gas constant (J mol⁻¹ K⁻¹)
        #   T         = temperature (K)
        rgas = 8.314  # J K-1 mol-1
        volume_m3 = self.lunchbox_volume / 1000.0  # litre to m3
        an = (delta_ppm_s * self.pressure * volume_m3) / (rgas * self.temp_k)

        return an # umol leaf s-1

    def read_and_update(self):
        current_time = time.time()

        if current_time - self.last_measure_time < self.measure_interval:
            # Not time to measure
            return None

        try:
            co2 = self.sensor.read_co2()

            if self.last_co2 is not None and abs(co2 - self.last_co2) < 0.01:
                co2 = self.last_co2  # avoid noisy updates, i.e. don't update
            else:
                self.last_co2 = co2

            # Re-apply pressure compensation here every measurement
            #self.sensor.set_pressure_reference(self.pressure)

        except Exception as e:
            print(f"Read error: {e}")
            if self.last_co2 is not None:
                co2 = self.last_co2
            else:
                return None

        self.last_measure_time = current_time
        self.co2_window.append(co2)
        self.time_window.append(current_time)

        if len(self.co2_window) < self.window_size:
            return None

        co2_array = np.array(self.co2_window)
        time_array = np.array(self.time_window)
        elapsed = time_array - time_array[0]
        elapsed = np.round(elapsed, 2)
        elapsed -= elapsed.mean()

        if self.smoothing and len(co2_array) >= self.window_size:
            co2_array_2 = medfilt(co2_array, kernel_size=5)
            co2_array_smooth = savgol_filter(co2_array_2, window_length=19,
                                             polyorder=2)
            # sampling frequency in Hz (e.g., 0.5 Hz if
            fs = 1 / self.measure_interval
            # remove high frequency noise
            cutoff = 0.1
            co2_array_filter = butter_lowpass_filter(co2_array_smooth, cutoff,
                                                     fs, order=5)
        else:
            co2_array_filter = co2_array

        # Rolling linear regression
        if self.rolling_regression:
            X = sm.add_constant(elapsed)
            #model = sm.OLS(co2_array_filter, X)

            # less sensitive to outliers
            model = sm.RLM(co2_array_filter, X, M=sm.robust.norms.HuberT())
            results = model.fit()
            slope = results.params[1]
            stderr = results.bse[1] if results.bse.size > 1 else 0
        else:
            (p, residuals, _, _, _) = np.polyfit(elapsed, co2_array_filter, 1,
                                                 full=True)
            slope = p[0]
            n = len(elapsed)
            if n > 2 and residuals.size > 0:
                residual_var = residuals[0] / (n - 2)
                x_var = np.var(elapsed, ddof=1)
                stderr = np.sqrt(residual_var / (n * x_var))
            else:
                stderr = 0

        slope_upper = slope + 1.96 * stderr
        slope_lower = slope - 1.96 * stderr

        anet_leaf = self.calc_anet(slope)
        anet_leaf_u = self.calc_anet(slope_upper)
        anet_leaf_l = self.calc_anet(slope_lower)

        if self.area_basis:
            anet_plot = -anet_leaf / self.leaf_area_m2
            anet_u = -anet_leaf_u / self.leaf_area_m2
            anet_l = -anet_leaf_l / self.leaf_area_m2
        else:
            anet_plot = -anet_leaf
            anet_u = -anet_leaf_u
            anet_l = -anet_leaf_l

        # Apply respiration correction if Anet < 0
        if anet_plot < 0:
            anet_plot += self.soil_resp_correction
            anet_u += self.soil_resp_correction
            anet_l += self.soil_resp_correction

        elapsed_min = (current_time - self.start_time) / 60

        return {
            "elapsed_min": elapsed_min,
            "co2": co2,
            "anet": anet_plot,
            "anet_lower": anet_l,
            "anet_upper": anet_u,
        }

    def close(self):
        self.sensor.close()


def butter_lowpass_filter(data, cutoff, fs, order=4):
    # suppress high-frequency sensor noise but keep long-period oscillations
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)

    return y


def butter_bandstop_filter(data, lowcut, highcut, fs, order=4):
    # Apply a Butterworth band-stop filter to removes periodic oscillations
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')

    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        print(f"Skipping bandstop filter (order={order}) -")
        print(f"input too short (len={len(data)} ≤ padlen={padlen})")
        return data

    return filtfilt(b, a, data)


def calc_volume_litres(width_cm, height_cm, length_cm):
    volume_cm3 = width_cm * height_cm * length_cm
    volume_litres = volume_cm3 / 1000

    return volume_litres


def calc_frustum_volume_litres(top_width_cm, base_width_cm, height_cm):
    """
    Calculate the volume in litres a pot with slopping sides
    """
    a = top_width_cm
    b = base_width_cm
    h = height_cm

    volume_cm3 = (h / 3) * (a**2 + a*b + b**2)
    volume_litres = volume_cm3 / 1000

    return volume_litres
