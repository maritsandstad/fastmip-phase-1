# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

import fair
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

# %%
print(fair.__version__)


# Here we need to rerun one normal scenario without natural forcings
# To get the correct temperature baseline for the idalised experiments



# %%
flat_em_levels = [7.5, 10, 20]
scenarios = []
for em_level in flat_em_levels:
    scenarios.append(f"esm-flat{em_level}")
    scenarios.append(f"esm-flat{em_level}_zec")
    scenarios.append(f"esm-flat{em_level}_cdr")
    scenarios.append(f"esm-flat{em_level}_rev")
    scenarios.append(f"esm-flat{em_level}_nz")

# %%
scenarios

# %%
n_scen = len(scenarios)
run_w_variability = True

# %%
species = ['CO2', 'CH4', 'N2O']
properties = {
    "CO2": {
        'type': 'co2',
        'input_mode': 'emissions',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False
    },
    "CH4": {
        'type': 'ch4',
        'input_mode': 'emissions',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False
    },
    "N2O": {
        'type': 'n2o',
        'input_mode': 'emissions',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False
    }
}

# %%
f = {}

# %%
solar_forcing = np.zeros(752)
volcanic_forcing = np.zeros(752)

df_configs = pd.read_csv(
    "../data/fair_parameters_1.6.0/"
    "calibrated_constrained_parameters.csv",
    index_col=0,
)

valid_all = df_configs.index
# %% [markdown]
# ## With internal variability

# %%
f = FAIR(ch4_method="Thornhill2021")

f.define_time(0, 320, 1)
f.define_scenarios(scenarios)
f.define_configs(valid_all)

f.define_species(species, properties)


f.define_configs(valid_all)
f.allocate()

# fill emissions: zero for non-CO2
f.emissions.loc[dict(specie="CH4")] = 0
f.emissions.loc[dict(specie="N2O")] = 0

# constant pre-industrial concentration for non-CO2 GHGs
f.concentration.loc[dict(specie='CH4')] = 808.2490285
f.concentration.loc[dict(specie='N2O')] = 273.021047

# %%
for scenario in scenarios:
    em_level = float(scenario.split("flat")[1].split("_")[0])
    type = scenario.split("flat")[1].split("_")[1] if "_" in scenario else "flat"
    if type == "flat":
        f.emissions.loc[dict(specie="CO2", scenario=scenario)] = em_level * 44.009 / 12.011
    else:
        f.emissions.loc[dict(specie="CO2", scenario=scenario, timepoints=np.arange(0.5, 100))] = em_level * 44.009 / 12.011
        if type == "zec":
            f.emissions.loc[dict(specie="CO2", scenario=scenario, timepoints=np.arange(100.5, 320))] = 0
        elif type == "cdr":
            f.emissions.loc[dict(specie="CO2", scenario=scenario, timepoints=np.arange(100.5, 200))] = em_level/10.*np.linspace(9.9, -9.9, 100)[:, None] * 44.009 / 12.011
            f.emissions.loc[dict(specie="CO2", scenario=scenario, timepoints=np.arange(200.5, 300))] = -em_level * 44.009 / 12.011
            f.emissions.loc[dict(specie="CO2", scenario=scenario, timepoints=np.arange(300.5, 320))] = 0
        elif type == "rev":
            f.emissions.loc[dict(specie="CO2", scenario=scenario, timepoints=np.arange(100.5, 200))] = 0
            f.emissions.loc[dict(specie="CO2", scenario=scenario, timepoints=np.arange(200.5, 300))] = -em_level * 44.009 / 12.011
            f.emissions.loc[dict(specie="CO2", scenario=scenario, timepoints=np.arange(300.5, 320))] = 0
        else:
            f.emissions.loc[dict(specie="CO2", scenario=scenario, timepoints=np.arange(100.5, 150))] = em_level/10.*np.linspace(9.9, -9.9, 50)[:, None] * 44.009 / 12.011
            f.emissions.loc[dict(specie="CO2", scenario=scenario, timepoints=np.arange(150.5, 320))] = 0

f.fill_species_configs()

# %%
# Climate response
print(df_configs.head())
if run_w_variability:
    fill(f.climate_configs['stochastic_run'], True)
else:
    fill(f.climate_configs['stochastic_run'], False)

# initial condition of CO2 concentration (but not baseline for forcing calculations)
fill(f.species_configs['baseline_concentration'], 284.3169988, specie='CO2')
fill(f.species_configs['baseline_concentration'], 808.2490285, specie='CH4')
fill(f.species_configs['baseline_concentration'], 273.021047, specie='N2O')
fill(f.species_configs["forcing_reference_concentration"], 284.3169988, specie='CO2')
fill(f.species_configs["forcing_reference_concentration"], 808.2490285, specie='CH4')
fill(f.species_configs["forcing_reference_concentration"], 273.021047, specie='N2O')

f.override_defaults(
    "../data/fair_parameters_1.6.0/calibrated_constrained_parameters.csv",
)

# set initial conditions
initialise(f.concentration, f.species_configs['baseline_concentration'])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.airborne_emissions, 0)
initialise(f.cumulative_emissions, 0)


f.run()

baseline_subtract = np.loadtxt('../output/baseline_subtract.txt')
# output[..., ivolc] = (
#     f.temperature.sel(layer=0) - f.temperature.sel(layer=0).weighted(weights).mean(dim="timebounds")
# ).sel(scenario='ssp245', timebounds=np.arange(0, 2102))
temperature_baseline_2004_2023 = (
    f.temperature.sel(layer=0) - baseline_subtract[0,:]
) + 1.05

# %%
temperature_baseline_2004_2023

# %%
pl.plot(temperature_baseline_2004_2023.median(dim="config"))

# %%
pl.plot((
    f.temperature.sel(layer=0) -
    f.temperature.sel(timebounds = np.arange(0, 321), layer=0).mean(dim="timebounds")
).median(dim="config"))

# %%
pl.plot(f.forcing_sum.median(dim="config"))

# %%
pl.plot(f.temperature.sel(layer=0, config=valid_all[0]))

# %%
pl.plot(temperature_baseline_2004_2023.sel(scenario=scenarios[0]));

# %%
[f'Climate Assessment|Surface Temperature (GSAT)|ensemble member {config} [fair-2.2.4 cal-1.6.0]' for config in valid_all]

# %%

def produce_variable_dataframe(variable_name, unit, data_array, plot = True):
    mi = []
    for scenario in scenarios:
        for config in valid_all:
            ix = (
                "idealised", 
                scenario, 
                'World', 
                variable_name,
                unit,
                config,
                'fair-2.2.4',
                '1.6.0-full',
            )
            mi.append(ix)
    index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
    temp_out_data = np.ones((321, 841*n_scen))*np.nan
    irow = 0
    for scenario in scenarios:
        # for config in valid_all:
        temp_out_data[:, irow:irow+841] = data_array.sel(scenario=scenario, config=valid_all, timebounds=np.arange(0, 321))
        irow = irow + 841
    if plot:
        plot_per_scenario(variable_name, unit, data_array)
    return pd.DataFrame(temp_out_data.T, index=index, columns=np.arange(0, 321))

def plot_per_scenario(variable_name, unit, data_array):
    fig, axs = pl.subplots(1, 1, figsize=(10, 5))
    for scenario in scenarios:
        axs.plot(data_array.sel(scenario=scenario, config=valid_all).median(dim="config"), label=scenario)
        axs.fill_between(
            np.arange(0, 321),
            data_array.sel(scenario=scenario, config=valid_all).quantile(0.05, dim="config"),
            data_array.sel(scenario=scenario, config=valid_all).quantile(0.95, dim="config"),
            alpha=0.2
        )
    axs.set_title(f"{variable_name} ({unit})")
    axs.set_xlabel("Time (years)")
    axs.set_ylabel(f"{variable_name} ({unit})")
    axs.legend()
    fig.tight_layout()
    fig.savefig(f"../output/{variable_name.replace('|', '_').replace(' ', '_')}.png", dpi=300)
    
# index = pd.MultiIndex.from_product(
#     [
#         scenarios,
#         ['World'],
#         [f'Climate Assessment|Surface Temperature (GSAT)|ensemble member {config} [fair-2.2.4 cal-1.6.0]' for config in valid_all],
#         ['K']
#     ],
#     names=["scenario", "region", "variable", "unit"]
# )

#
# The next few cells get repeated with the variables of interest

# %%
temp_out = produce_variable_dataframe('Climate Assessment|Surface Temperature (GSAT)', 'K', temperature_baseline_2004_2023)
forcing_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing', 'W/m2', f.forcing_sum)
toa_out = produce_variable_dataframe('Climate Assessment|Top of Atmosphere Energy Imbalance', 'W/m2', f.toa_imbalance)


# %%
greenhouse_gases = [
    'CO2',
    'CH4',
    'N2O',
]
forcing_ghg = f.forcing.sel(specie=greenhouse_gases).sum(dim='specie')
forcing_ghg_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing|Anthropogenic|Greenhouse Gases', 'W/m2', forcing_ghg)


# %%
conc_co2 = f.concentration.sel(specie='CO2')
concentration_co2_out = produce_variable_dataframe('Climate Assessment|Concentration|CO2', 'ppm', conc_co2)
forcing_other_out = forcing_out - forcing_ghg_out 
# %%
forcing_other = f.forcing_sum - forcing_ghg
forc_other_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing|Anthropogenic|Other', 'W/m2', forcing_other)


# %%
data_out = pd.concat(
    (
        temp_out, 
        toa_out,
        forcing_out, 
        forcing_ghg_out, 
        forcing_other_out,
        concentration_co2_out
    )
)

# %%
data_out

# %%
os.makedirs('../output', exist_ok=True)

# %%
if run_w_variability:
    ending = "full"
else:
    ending = "forced"
data_out.to_csv(f'../output/climate_assessment_flat10_{ending}.csv')