import csv
import time
import datetime
from collections import deque
import numpy as np
import statsmodels.api as sm

from xensiv_pas_co2_sensor import CO2Sensor, MEAS_RATE_MIN_S

# Box and pot geometry
BOX_VOLUME_L = 0.5
POT_TOP_CM = 5.0     # square pot, top edge
POT_BASE_CM = 3.4    # square pot, base edge
POT_HEIGHT_CM = 5.3
PRESSURE_PA = 101325.0
RGAS = 8.314  # J K-1 mol-1


class LunchboxLogger:
    def __init__(self, port, baud, lunchbox_volume, temp_c, leaf_area_cm2,
                 window_size, measure_interval=MEAS_RATE_MIN_S, timeout=1.0,
                 robust=True, area_basis=True, soil_resp_correction=0.0,
                 soil_area_m2=None, pressure_pa=PRESSURE_PA, csv_path=None,
                 warmup_readings=4):

        if window_size < 5:
            raise ValueError("window_size must be ≥ 5 samples")
        if soil_resp_correction < 0:
            raise ValueError("soil_resp_correction is the soil CO2 efflux and "
                             "must be ≥ 0 (μmol m⁻² soil s⁻¹)")

        self.temp_k = temp_c + 273.15
        self.pressure = pressure_pa
        self.leaf_area_m2 = leaf_area_cm2 / 10000.0
        self.lunchbox_volume = lunchbox_volume
        self.window_size = window_size
        self.measure_interval = measure_interval
        self.area_basis = area_basis
        self.robust = robust
        # The first few readings after a reset jump around, drop them
        self.warmup_readings = warmup_readings

        # Soil efflux adds CO2 to the box all the time, so what we measure is
        # leaf uptake minus soil efflux. Convert the per-soil-area correction
        # to a whole-box flux (μmol s-1) and add it back on every reading.
        if soil_area_m2 is None:
            soil_area_m2 = (POT_TOP_CM / 100.0) ** 2
        self.soil_flux_umol_s = soil_resp_correction * soil_area_m2

        # Data buffers
        self.co2_window = deque(maxlen=window_size)
        self.time_window = deque(maxlen=window_size)

        self.csv_file = None
        if csv_path:
            self.csv_file = open(csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["time", "elapsed_s", "co2_ppm", "anet",
                                      "anet_lower", "anet_upper"])

        # Setup sensor
        self.sensor = CO2Sensor(port, baud, timeout)
        try:
            self.sensor.reset_sensor()
            self.sensor.set_pressure_reference(self.pressure)
            self.sensor.arm_sensor(rate_seconds=self.measure_interval)
        except Exception as e:
            print(f"Failed to arm sensor: {e}")
            self.sensor.close()
            raise

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
        volume_m3 = self.lunchbox_volume / 1000.0  # litre to m3
        an = (delta_ppm_s * self.pressure * volume_m3) / (RGAS * self.temp_k)

        return an # umol leaf s-1

    def read_and_update(self):
        """
        Poll the sensor. Returns None if there is no new reading yet,
        otherwise a dict; "anet" is None until the window has filled.
        """
        try:
            if not self.sensor.is_data_ready():
                return None
            co2 = self.sensor.read_co2()
        except Exception as e:
            # Skip the sample rather than inventing one, a repeated value
            # would drag the slope towards zero
            print(f"Read error: {e}")
            return None

        if self.warmup_readings > 0:
            self.warmup_readings -= 1
            return None

        current_time = time.time()
        self.co2_window.append(co2)
        self.time_window.append(current_time)

        result = {
            "elapsed_min": (current_time - self.start_time) / 60,
            "co2": co2,
            "n": len(self.co2_window),
            "anet": None,
            "anet_lower": None,
            "anet_upper": None,
        }

        if len(self.co2_window) == self.window_size:
            slope, stderr = fit_slope(np.array(self.time_window),
                                      np.array(self.co2_window), self.robust)

            # CO2 falling in the box = uptake, so flip the sign
            fluxes = [-self.calc_anet(s) + self.soil_flux_umol_s for s in
                      (slope, slope + 1.96 * stderr, slope - 1.96 * stderr)]
            if self.area_basis:
                fluxes = [f / self.leaf_area_m2 for f in fluxes]
            anet, anet_l, anet_u = fluxes

            result.update(anet=anet, anet_lower=anet_l, anet_upper=anet_u)

        if self.csv_file:
            now = datetime.datetime.fromtimestamp(current_time)
            self.csv_writer.writerow([now.isoformat(timespec="seconds"),
                                      round(current_time - self.start_time, 1),
                                      co2, result["anet"],
                                      result["anet_lower"],
                                      result["anet_upper"]])
            self.csv_file.flush()

        return result

    def close(self):
        try:
            self.sensor.set_idle()
        except Exception:
            pass
        self.sensor.close()
        if self.csv_file:
            self.csv_file.close()


def fit_slope(time_s, co2_ppm, robust=True):
    """
    Linear fit of CO2 (ppm) against time (s). Returns slope (ppm s-1) and its
    standard error. The robust (Huber) fit down-weights the occasional spike,
    so the raw readings can be used without pre-smoothing, which would make
    the standard error meaningless.
    """
    elapsed = time_s - time_s.mean()
    X = sm.add_constant(elapsed)
    if robust:
        results = sm.RLM(co2_ppm, X, M=sm.robust.norms.HuberT()).fit()
    else:
        results = sm.OLS(co2_ppm, X).fit()

    return results.params[1], results.bse[1]


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


def air_volume_litres(no_plant_pot=False):
    """Air volume in the closed box, i.e. box minus the pot."""
    if no_plant_pot:
        return BOX_VOLUME_L
    return BOX_VOLUME_L - calc_frustum_volume_litres(POT_TOP_CM, POT_BASE_CM,
                                                     POT_HEIGHT_CM)
