"""Resubmission model; SI at the boundary, explicitly scaled solver coordinates."""
from dataclasses import dataclass, asdict
import hashlib,json
import numpy as np
from .energy import Battery

RATE = 1e9  # One solver unit = 10^9 bit/s; all saved rates are SI.
POWER = 100.0  # One resource coordinate = 100 W.

@dataclass(frozen=True)
class Config:
    seed:int=1000
    planes:int=24
    per_plane:int=24
    orbits:int=3
    slots_per_orbit:int=96
    flows:int=288
    paths:int=4
    routing:str='hybrid'
    medium:str='optical'
    interference_db:float=-300.
    preset:str='tight'
    beta_deg:float=0.
    load:float=.65
    traffic:str='mmpp'
    initial_soc:float=.95
    base_w:float=55.
    solar_w:float=500.
    alpha:float=.01
    lam:float=.1
    epsilon:float=.1
    rho:float=.02
    utility_s_bps:float=1e9
    max_iterations:int=200
    tolerance:float=1e-3
    guard_soc:float=.3
    artificial_eclipse:float=0.
    dt_override_s:float=0.
    delay_s:float=0.
    substep_s:float=0.
    demand_mode:str='cities'
    heterogeneous:bool=False
    capacity_j:float=100e6
    dod_max:float=.30
    cut_soc:float=.10
    sleep_w:float=2.
    acquisition_s:float=20.
    kappa_off:float=1.
    xi_wake:float=.3
    minimum_hold_slots:int=2
    wake_slack_w:float=5.
    rho_energy:float=0.
    dual_eta:float=.1
    dual_warm:str='zero'
    lyapunov_lambda:float=.5
    lyapunov_theta:float=.6
    utilization_sleep_threshold:float=.05
    rf_capture_power:bool=True
    acquisition_carry:bool=True

    def key(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:16]

    @property
    def battery(self):
        return Battery(self.capacity_j,(1-self.dod_max)*self.capacity_j,.95,.95)

@dataclass(frozen=True)
class Terminal:
    medium:str='optical'
    maximum_range_km:float=6000.
    idle_w:float=25.
    max_electrical_w:float=60.
    max_optical_w:float=2.
    eta_wp:float=.2
    e_bit_j:float=2e-9
    wavelength_m:float=1550e-9
    aperture_m:float=.08
    beam_half_angle_rad:float=10e-6
    photons_per_bit:float=1000.
    design_rate_bps:float=10e9
    rf_bandwidth_hz:float=500e6
    rf_frequency_hz:float=26e9
    rf_gain_db:float=30.
    rf_temperature_k:float=318.
    rf_eta_pa:float=.35
    rf_max_radiated_w:float=10.
    interference_db:float=-300.

    def optical_j_bit(self,distance_km):
        spot=np.asarray(distance_km)*1000*self.beam_half_angle_rad
        capture=-np.expm1(-2*(self.aperture_m/2/spot)**2)
        return self.photons_per_bit*6.62607015e-34*299792458/self.wavelength_m/capture

    def kappa(self,distance_km):
        path=(4*np.pi*np.asarray(distance_km)*1000*self.rf_frequency_hz/299792458.)**2
        return 1.380649e-23*self.rf_temperature_k*self.rf_bandwidth_hz*path/10**(2*self.rf_gain_db/10)*(1+10**(self.interference_db/10))

    def capacity(self,distance_km):
        d=np.asarray(distance_km)
        if self.medium=='optical':
            optical=self.optical_j_bit(d)
            rate=np.minimum(self.design_rate_bps,np.minimum(self.max_optical_w/optical,(self.max_electrical_w-self.idle_w)/(optical/self.eta_wp+self.e_bit_j)))
        else:
            radiated=min(self.rf_max_radiated_w,(self.max_electrical_w-self.idle_w)*self.rf_eta_pa)
            rate=self.rf_bandwidth_hz*np.log2(1+radiated/self.kappa(d))
        return np.where(d<=self.maximum_range_km,rate,0.)

    def increment(self,rate_bps,distance_km):
        r=np.asarray(rate_bps)
        if self.medium=='optical': return (self.optical_j_bit(distance_km)/self.eta_wp+self.e_bit_j)*r
        return self.kappa(distance_km)*np.expm1(np.log(2)*r/self.rf_bandwidth_hz)/self.rf_eta_pa

    def derivative(self,rate_bps,distance_km):
        if self.medium=='optical': return self.optical_j_bit(distance_km)/self.eta_wp+self.e_bit_j
        return self.kappa(distance_km)*np.log(2)/self.rf_bandwidth_hz*np.exp(np.log(2)*np.asarray(rate_bps)/self.rf_bandwidth_hz)/self.rf_eta_pa

    def slot_increment(self,rate_bps,distance_km,capture=None):
        """Slot-average RF power at the within-slot transmission rate."""
        if self.medium=='optical' or capture is None:
            return self.increment(rate_bps,distance_km)
        capture=np.asarray(capture)
        instant=np.divide(rate_bps,capture,out=np.zeros_like(np.asarray(rate_bps),dtype=float),where=capture>0)
        return capture*self.increment(instant,distance_km)

    def slot_derivative(self,rate_bps,distance_km,capture=None):
        if self.medium=='optical' or capture is None:
            return self.derivative(rate_bps,distance_km)
        instant=np.divide(rate_bps,capture,out=np.zeros_like(np.asarray(rate_bps),dtype=float),where=np.asarray(capture)>0)
        return self.derivative(instant,distance_km)
