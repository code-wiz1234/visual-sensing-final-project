"""Commercial DVS and CIS sensor database with real-world specs and pricing.

Sources:
  DVS:
    Lichtsteiner 2008 — "A 128x128 120dB 15us Latency Asynchronous Temporal
        Contrast Vision Sensor", JSSC 2008, Table 1
        https://doi.org/10.1109/JSSC.2007.914337

    DAVIS346 — iniVation DAVIS346 datasheet rev 2.0 (2019)
        https://inivation.com/wp-content/uploads/2019/08/DAVIS346.pdf

    Samsung DVS-Gen3.1 — "A 640x480 Dynamic Vision Sensor with a 9um Pixel
        and 300Meps Address-Event Representation", ISSCC 2020
        https://www.researchgate.net/publication/314296009_41_A_640480_dynamic_vision_sensor_with_a_9m_pixel_and_300Meps_address-event_representation

    Prophesee IMX636 — Sony/Prophesee IMX636 EVK4 datasheet (2022)
        https://www.prophesee.ai/wp-content/uploads/2023/01/EVK4-HD-Prophesee-Evaluation-Kit-Camera-Manual-1.1.pdf

  CIS:
    OV7251 (OmniVision) — VGA global shutter, datasheet 2017
        https://www.ovt.com/products/ov7251/

    IMX327 (Sony) — 1080p starlight, datasheet 2018
        https://www.1stvision.com/cameras/sensor_specs/IMX327.pdf

    AR0234 (ON Semi) — automotive 1080p global shutter, datasheet 2020
        https://www.onsemi.com/parametrics/AR0234CS/create-overview-pdf

    IMX462 (Sony) — 1080p starlight high-fps, datasheet 2021
        https://www.sony-semicon.com/files/62/flyer_security/IMX462LQR_LQR1_Flyer.pdf

Prices are approximate module/dev-kit street prices (2024-2025). Bare die
prices are 2-10x lower but not publicly available for most sensors.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DVSSensor:
    name: str
    resolution: tuple[int, int]  # (width, height)
    p_static_mw: float  # always-on power (mW)
    e_per_event_nj: float  # energy per event (nJ)
    pixel_latency_us: float  # per-pixel response time (us)
    max_event_rate_mevps: float  # refractory cap (M events/sec)
    supply_v: float  # supply voltage
    price_usd: str  # approximate price or range
    notes: str = ""

    def power_mw(self, event_rate: float) -> float:
        """Total power given a measured event rate (events/sec)."""
        eff_rate = min(event_rate, self.max_event_rate_mevps * 1e6)
        dynamic = eff_rate * self.e_per_event_nj * 1e-9 * 1e3  # nJ -> mW
        return self.p_static_mw + dynamic

    def position_error_px(self, velocity_pxps: float) -> float:
        return velocity_pxps * self.pixel_latency_us * 1e-6


@dataclass
class CISSensor:
    name: str
    resolution: tuple[int, int]
    max_fps: float
    power_at_max_fps_mw: float  # total power at max FPS
    power_idle_mw: float  # standby / low-fps power
    adc_bits: int
    supply_v: float
    price_usd: str
    notes: str = ""

    def power_mw(self, required_fps: float) -> float:
        """Interpolate power between idle and max-fps."""
        if required_fps <= 0:
            return self.power_idle_mw
        frac = min(required_fps / self.max_fps, 1.0)
        return self.power_idle_mw + frac * (
            self.power_at_max_fps_mw - self.power_idle_mw
        )


# ── DVS sensors ────────────────────────────────────────────────────────

DVS_SENSORS = [
    DVSSensor(
        name="Lichtsteiner 2008",
        resolution=(128, 128),
        p_static_mw=19.14,  # (0.3 + 5.5 mA) * 3.3V
        e_per_event_nj=4.95,  # 1.5mA * 3.3V / 1Mev/s
        pixel_latency_us=15.0,  # paper §V
        max_event_rate_mevps=1.0,  # per 128x128 array
        supply_v=3.3,
        price_usd="N/A (research)",
        notes="Original DVS 128x128. https://doi.org/10.1109/JSSC.2007.914337",
    ),
    DVSSensor(
        name="DAVIS346",
        resolution=(346, 260),
        p_static_mw=10.0,  # datasheet: DVS 10-30mW activity dependent, 10mW idle
        e_per_event_nj=0.49,  # (30mW-10mW) peak dynamic / 12Mev/s bandwidth
        pixel_latency_us=20.0,  # datasheet: "Min. latency ~ 20 us"
        max_event_rate_mevps=12.0,  # datasheet: 12 MEvents/second bandwidth
        supply_v=3.3,  # chip: 1.8V + 3.3V, camera: 5V USB
        price_usd="~$3,000 (dev kit)",
        notes="iniVation hybrid frames+events, 18.5um pixel, 0.18um CIS. "
              "https://inivation.com/wp-content/uploads/2019/08/DAVIS346.pdf",
    ),
    DVSSensor(
        name="Samsung DVS-Gen3.1",
        resolution=(640, 480),
        p_static_mw=10.0,  # ISSCC 2020: ~10mW static
        e_per_event_nj=0.20,  # advanced 40nm process
        pixel_latency_us=5.0,  # ISSCC: <10us typ
        max_event_rate_mevps=50.0,
        supply_v=1.8,
        price_usd="~$100-300 (module)",
        notes="Samsung 640x480 40nm. https://doi.org/10.1109/ISSCC19947.2020.9063149",
    ),
    DVSSensor(
        name="Prophesee IMX636",
        resolution=(1280, 720),
        p_static_mw=32.0,  # sensor chip idle (EVK4 total: 500mW typ incl. USB bridge)
        e_per_event_nj=0.10,  # Sony stacked BSI
        pixel_latency_us=220.0,  # EVK4 datasheet: "Pixel Typical Latency 220us" at 25% contrast
        max_event_rate_mevps=3000.0,  # EVK4 datasheet: "Maximum ReadOut throughput 3 Gevents/s"
        supply_v=1.8,
        price_usd="~$3,500 (EVK4 kit)",
        notes="Sony/Prophesee 1280x720, 4.86um pixel, stacked BSI, 1/2.5in format. "
              "https://www.prophesee.ai/wp-content/uploads/2023/01/"
              "EVK4-HD-Prophesee-Evaluation-Kit-Camera-Manual-1.1.pdf",
    ),
]

# ── CIS sensors ────────────────────────────────────────────────────────

CIS_SENSORS = [
    CISSensor(
        name="OV7251 (OmniVision)",
        resolution=(640, 480),
        max_fps=120.0,
        power_at_max_fps_mw=40.0,  # datasheet: 40mW at 640x480/120fps
        power_idle_mw=2.0,
        adc_bits=10,
        supply_v=1.8,
        price_usd="~$3-5",
        notes="VGA global shutter, drone/AR. https://www.ovt.com/products/ov7251/",
    ),
    CISSensor(
        name="IMX327 (Sony)",
        resolution=(1920, 1080),
        max_fps=60.0,
        power_at_max_fps_mw=300.0,
        power_idle_mw=15.0,
        adc_bits=12,
        supply_v=2.9,
        price_usd="~$15-25",
        notes="Starvis 1080p, surveillance. https://www.sony-semicon.com/files/62/pdf/p-13_IMX327LQR_Flyer.pdf",
    ),
    CISSensor(
        name="AR0234 (ON Semi)",
        resolution=(1920, 1200),
        max_fps=120.0,
        power_at_max_fps_mw=250.0,
        power_idle_mw=10.0,
        adc_bits=10,
        supply_v=2.8,
        price_usd="~$20-40",
        notes="Automotive global shutter, ADAS. https://www.onsemi.com/products/sensors/image-sensors/ar0234cs",
    ),
    CISSensor(
        name="IMX462 (Sony)",
        resolution=(1920, 1080),
        max_fps=120.0,
        power_at_max_fps_mw=310.0,
        power_idle_mw=12.0,
        adc_bits=12,
        supply_v=2.9,
        price_usd="~$20-30",
        notes="Starvis 2 high-fps 1080p. https://www.sony-semicon.com/files/62/pdf/p-13_IMX462LQR_Flyer.pdf",
    ),
]


def print_sensor_table():
    """Print a formatted summary of all sensors."""
    print("=" * 100)
    print("DVS SENSORS")
    print("=" * 100)
    print(
        f"{'Name':<25s} {'Resolution':>10s} {'P_static':>9s} {'E/event':>8s} "
        f"{'Latency':>8s} {'Max rate':>10s} {'Price':>20s}"
    )
    print("-" * 100)
    for sensor in DVS_SENSORS:
        res = f"{sensor.resolution[0]}x{sensor.resolution[1]}"
        print(
            f"{sensor.name:<25s} {res:>10s} {sensor.p_static_mw:>7.1f}mW "
            f"{sensor.e_per_event_nj:>6.2f}nJ {sensor.pixel_latency_us:>6.1f}us "
            f"{sensor.max_event_rate_mevps:>7.0f}Mev/s {sensor.price_usd:>20s}"
        )

    print()
    print("=" * 100)
    print("CIS SENSORS")
    print("=" * 100)
    print(
        f"{'Name':<25s} {'Resolution':>10s} {'MaxFPS':>7s} {'Power@max':>10s} "
        f"{'Idle':>7s} {'ADC':>4s} {'Price':>20s}"
    )
    print("-" * 100)
    for sensor in CIS_SENSORS:
        res = f"{sensor.resolution[0]}x{sensor.resolution[1]}"
        print(
            f"{sensor.name:<25s} {res:>10s} {sensor.max_fps:>5.0f}fp "
            f"{sensor.power_at_max_fps_mw:>8.0f}mW {sensor.power_idle_mw:>5.0f}mW "
            f"{sensor.adc_bits:>3d}b {sensor.price_usd:>20s}"
        )


if __name__ == "__main__":
    print_sensor_table()
