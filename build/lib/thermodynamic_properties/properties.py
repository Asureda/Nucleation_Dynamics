import numpy as np
import pint

ureg = pint.UnitRegistry()

class Lithium:
    def __init__(self):
        pass

    def density(self, T, correlation="Jeppson78"):
        correlations = {
            "Jeppson78": {
                "expression": "0.515-1.01e-4*((T-273.15)-200)",
                "bounds": "200-1600°C",
                "units": ureg('g/cm^3')
            }
        }
        if correlation in correlations:
            return eval(correlations[correlation]["expression"]) * correlations[correlation]["units"]
        return "Correlación desconocida"

    def specific_heat(self, T, correlation="Jeppson78"):
        correlations = {
            "Jeppson78": {
                "expression": "1.14 -5.06e-4*(T-273.15) +4.45e-7*(T-273.15)**2" if 200 <= T <= 600 else "0.995",
                "bounds": "200-600°C, 0.995 para 650-1000°C"
            }
        }
        if correlation in correlations:
            print(f"Rango de {correlation}: {correlations[correlation]['bounds']}")
            return eval(correlations[correlation]["expression"]) * ureg('cal/g/degC')
        return "Correlación desconocida"

    def viscosity(self, T, correlation="Jeppson78"):
        correlations = {
            "Jeppson78": {
                "expression": "10**(-1.3380+726.07/T)",
                "bounds": "873.15-1473.15 K"
            },
            "Buxbaum82": {
                "expression": "10**(1.4936-0.7368*np.log10(T)+109.95/T)",
                "bounds": "458.15-1273.15 K"
            }
        }
        if correlation in correlations:
            print(f"Rango de {correlation}: {correlations[correlation]['bounds']}")
            return eval(correlations[correlation]["expression"]) * ureg('centipoise')
        return "Correlación desconocida"
    
    def thermal_conductivity(self, T, correlation="Davison"):
        correlations = {
            "Davison": {
                "expression": "21.874 + 5.6255e-2*T -1.8325e-5*T**2",
                "bounds": "453.7-1610 K",
                "units": ureg('W/m/K')
            },
            "Jeppson78": {
                "expression": "(40.09 + 1.902e-2*(T-273.15)) * 418.4",  # Convertir a W/m/K
                "bounds": "250-1100°C",
                "units": ureg('W/m/K')
            },
            "Mills97": {
                "expression": "43 + 2e-2*(T_k - 171)",
                "bounds": "444.15-1373.15 K",
                "units": ureg('W/m/K')
            }
        }
        if correlation in correlations:
            print(f"Rango de {correlation}: {correlations[correlation]['bounds']}")
            return eval(correlations[correlation]["expression"]) * correlations[correlation]["units"]
        return "Correlación desconocida"

    def magnetic_permeability(self, correlation="LandoltBornstein86_92"):
        correlations = {
            "LandoltBornstein86_92": {
                "expression": "14.2e-6",
                "bounds": "285-300 K",
                "units": ureg('cm^3/mol')
            }
        }
        if correlation in correlations:
            print(f"Rango de {correlation}: {correlations[correlation]['bounds']}")
            return eval(correlations[correlation]["expression"]) * correlations[correlation]["units"]
        return "Correlación desconocida"

    def electrical_resistivity(self, T, correlation="Calaway82"):
        correlations = {
            "Calaway82": {
                "expression": "0.1891+3.680e-4*(T-273.15)-1.639e-7*(T-273.15)**2+7.507e-11*(T-273.15)**3",
                "bounds": "181-1000°C",
                "units": ureg('microohm*m')
            }
        }
        if correlation in correlations:
            print(f"Rango de {correlation}: {correlations[correlation]['bounds']}")
            return eval(correlations[correlation]["expression"]) * correlations[correlation]["units"]
        return "Correlación desconocida"

    def fusion_temperature(self, correlation="Jeppson78"):
        correlations = {
            "Jeppson78": {
                "expression": "453.69",
                "bounds": "1e5-2e5 Pa",
                "units": ureg('K')
            }
        }
        if correlation in correlations:
            print(f"Rango de {correlation}: {correlations[correlation]['bounds']}")
            return eval(correlations[correlation]["expression"]) * correlations[correlation]["units"]
        return "Correlación desconocida"

    def surface_tension(self, T, correlation="Jeppson78"):
        correlations = {
            "Jeppson78": {
                "expression": "0.16*(3550 - T) - 95",
                "bounds": "200-1300 degrees",
                "units": ureg('dyne/cm')
            }
        }
        if correlation in correlations:
            print(f"Rango de {correlation}: {correlations[correlation]['bounds']}")
            return eval(correlations[correlation]["expression"]) * correlations[correlation]["units"]
        return "Correlación desconocida"


    def vapor_pressure(self, T, correlation="Jeppson78"):
        correlations = {
            "Jeppson78": {
                "expression": "10**(8.00-8143/T)",
                "bounds": "700-1400 K",
                "units": ureg('mmHg')
            }
        }
        if correlation in correlations:
            print(f"Rango de {correlation}: {correlations[correlation]['bounds']}")
            return eval(correlations[correlation]["expression"]) * correlations[correlation]["units"]
        return "Correlación desconocida"
    
    def enthalpy(self, T, correlation="Jeppson78"):
        correlations = {
            "Jeppson78": {
                "expression": "-5.075 + 1.0008*T -5.173*1e-3*T**-1",
                "bounds": "500 -1300 degrees",
                "units": ureg('cal/g')
            }
        }
        if correlation in correlations:
            print(f"Rango de {correlation}: {correlations[correlation]['bounds']}")
            return eval(correlations[correlation]["expression"]) * correlations[correlation]["units"]
        return 
    
    
    def diffusivity_tritium(self, T, correlation="Buxbaum82"):
        correlations = {
            "Buxbaum82": {
                "expression": "10**(-9.038+1.737*np.log10(T)-110/T)",
                "bounds": "800-905 K",
                "units": ureg('cm^2/s')
            }
        }
        if correlation in correlations:
            return eval(correlations[correlation]["expression"]) * correlations[correlation]["units"]
        return "Correlación desconocida"

    def solubility_tritium(self, T, correlation="Smith1979"):
        correlations = {
            "Smith75": {
                "expression": "-257.5+0.44*(T-273.15)",
                "bounds": "700-1400°C",
                "units": ureg('mmHg**0.5')
            },
            "Veleckis_fitted_Katsuta1976": {
                "expression": "5527.822*np.exp(-3648.173/(T-273.15))",
                "units": ureg('mmHg**0.5')
            },
            "Katsuta1976": {
                "expression": "10**(3.86-2320/T)",
                "units": ureg('mmHg**0.5')
            },
            "Veleckis1973": {
                "expression": "4412.7934*np.exp(-3465.8476/(T-273.15))",
                "units": ureg('mmHg**0.5')
            },
            "Katsuta1976_fitted": {
                "expression": "1768.665*np.exp(-2880.666/(T-273.15))",
                "units": ureg('mmHg**0.5')
            },
            "Smith": {
                "expression": "3552.287*np.exp(-2943.171/(T-273.15))",
                "units": ureg('mmHg**0.5')
            },
            "Smith1979": {
                "expression": "np.exp(9.226-5085/T)",
                "units": ureg('torr**0.5')
            }
        }
        if correlation in correlations:
            return eval(correlations[correlation]["expression"])*ureg.Quantity(1.0,"torr**0.5")
        return "Correlación desconocida"

    def solubility_deuterium(self, T, correlation="Smith1979"):
        correlations = {
            "Smith1979": {
                "expression": "np.exp(9.515-5644/T)",
                "units": ureg('torr**0.5')
            }
        }
        if correlation in correlations:
            return eval(correlations[correlation]["expression"])*ureg.Quantity(1.0,"torr**0.5")
        return "Correlación desconocida"
    
    def solubility_hydrogen(self, T, correlation="Smith1979"):
        correlations = {
            "Smith1979": {
                "expression": "np.exp(9.842-6242/T)",
                "units": ureg('torr**0.5')
            }
        }
        if correlation in correlations:
            return eval(correlations[correlation]["expression"])*ureg.Quantity(1.0,"torr**0.5")
        return "Correlación desconocida"

def inv_T(T):
    return 1000/T


