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
import os,sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import xarray as xr

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

# %%
# historical = pd.read_csv('../data/emissions/historical_emissions_1750-2023_cmip7.csv')
run_choice = "WIEMIP"
run_w_variability = True

startyear = 1750
endyear = 2501
# %%
# historical.loc[historical.variable=='Halon-1202'].index

# %%
# historical = historical.drop(index=historical.loc[historical.variable=='Halon-1202'].index)

# %%
# emitted_species = list(historical.variable.values)
polmip_wiemip_dict ={
    "PolMIP": {
        "em_file" : "../data/emissions/emissions_1750-2500_vlcf-ctap_test.csv",
        "scenarios" : ["VL-CF", "VL-CF-CTAP"]
    },
    "WIEMIP":{
        "em_file" : "../data/emissions/emissions_1750-2500_WIEMIP_v1.6.0_4.csv",
        "scenarios" : ["HL-CTAP", "ML-CTAP", "VL-CTAP"]
    }
}

# %%
future = pd.read_csv(polmip_wiemip_dict[run_choice]["em_file"])
print(future.columns)
future.rename(columns=lambda col: col.replace(".5", ".0"), inplace=True)
print(future.columns)
#sys.exit(4)

# %%
future.loc[future.variable=='Halon-1202'].index

# %%
future = future.drop(index=future.loc[future.variable=='Halon-1202'].index)

# %%
if "CO2|Gross Positive Emissions" in future.variable.values:
    future = future.drop(index=future.loc[future.variable=='CO2|Gross Positive Emissions'].index)

# %%
if "CO2|Gross Removals" in future.variable.values:
    future = future.drop(index=future.loc[future.variable=='CO2|Gross Removals'].index)

# %%
emitted_species = list(future.variable.unique())
print(emitted_species)

# %%
scenarios = list(future.scenario.unique())

# %%
print(scenarios)
scenarios = polmip_wiemip_dict[run_choice]["scenarios"]
print("Updated scenarios based on run choice:", scenarios)



# %%
n_scen = len(scenarios)

# %%
df_solar = pd.read_csv('../data/forcing/solar_forcing_timebounds_cmip7.csv', index_col='year')
df_volcanic = pd.read_csv('../data/forcing/volcanic_forcing_timebounds_cmip7.csv', index_col='Year')
df_irrigation = pd.read_csv('../data/forcing/irrigation_forcing_timebounds_cmip7.csv', index_col=0)
df_landuse = pd.read_csv('../data/forcing/land_use_forcing_timebounds_cmip7.csv', index_col=0)

# %%
solar_forcing = np.zeros(752)
volcanic_forcing = np.zeros(752)

# %%
scenarios_mapping_forcing = {
    'VL-CF': 'VL',
    'VL-CF-CTAP': 'VL',
    'VL-CTAP': 'VL',
    'HL-CTAP': 'HL',
    'ML-CTAP': 'ML',
}

# %%
volcanic_forcing = df_volcanic["volcanic_erf_rel_1850-2021"].loc[startyear:endyear +1].values
solar_forcing = df_solar["solar_erf_rel_1850-2019"].loc[startyear:endyear +1].values

print("Loaded in data, now ready to define the FaIR runs")

# %% [markdown]
# ## With internal variability

# %%
f = FAIR(ch4_method="Thornhill2021")
print("Set up the FaIR object, now defining the time and scenarios")
f.define_time(startyear, endyear, 1)
f.define_scenarios(scenarios)

species, properties = read_properties(
    "../data/fair_parameters_1.6.0/"
    "species_configs_properties.csv",
)
print("Defined the time and scenarios, now defining the species and properties")
f.define_species(species, properties)
print("Defined the species and properties, now reading in the calibrated parameters and defining the configs")
df_configs = pd.read_csv(
    "../data/fair_parameters_1.6.0/"
    "calibrated_constrained_parameters.csv",
    index_col=0,
)

valid_all = df_configs.index
print(valid_all.shape)

print("Read in the calibrated parameters and defined the configs, now allocating memory for the model runs")
f.define_configs(valid_all)
f.allocate()
print("FaIR is set up, now filling in the emissions and forcing data")

# %%
for scenario in scenarios:
    for specie in emitted_species:
        # f.emissions.loc[dict(timepoints=np.arange(startyear.5, 2024), scenario=scenario, specie=specie)] = (
        #     historical.loc[historical.variable==specie, 'startyear':].T
        # )
        f.emissions.loc[dict(timepoints=np.arange(startyear+0.5, endyear), scenario=scenario, specie=specie)] = (
            future.loc[(future.variable==specie) & (future.scenario==scenario), '1750.0':].T
        )
    f.forcing.loc[dict(scenario=scenario, specie='Land use')] = (
        df_landuse.loc[startyear:endyear, scenarios_mapping_forcing[scenario]].values[:, None] * df_configs["forcing_scale[Land use]"].values.squeeze()
    )
    f.forcing.loc[dict(scenario=scenario, specie='Irrigation')] = (
        df_irrigation.loc[startyear:endyear, scenarios_mapping_forcing[scenario]].values[:, None] * df_configs["forcing_scale[Irrigation]"].values.squeeze()
    )

# %%
fill(
    f.forcing,
    volcanic_forcing[:, None, None] * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
    specie="Volcanic",
)
fill(
    f.forcing,
    solar_forcing[:, None, None] * df_configs["forcing_scale[Solar]"].values.squeeze(),
    specie="Solar",
)

# %%
f.fill_species_configs(
    "../data/fair_parameters_1.6.0/species_configs_properties.csv",
)

if run_w_variability:
    fill(f.climate_configs['stochastic_run'], True)
else:
    fill(f.climate_configs['stochastic_run'], False)

f.override_defaults(
    "../data/fair_parameters_1.6.0/calibrated_constrained_parameters.csv",
)

# %%
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

print("FaIR is set up, now running the model - this will take a few minutes")

# %%
f.run()

print("Ran FaIR, now doing some post-processing to get the output in the right format")
# %%
weights = np.zeros((752, n_scen, 841))
weights[100, :, :] = 0.5
weights[101:151, :, :] = 1
weights[151, :, :] = 0.5
weights = xr.DataArray(
    weights, 
    dims=f.temperature.sel(layer=0).dims, 
    coords=f.temperature.sel(layer=0).coords
)
# output[..., ivolc] = (
#     f.temperature.sel(layer=0) - f.temperature.sel(layer=0).weighted(weights).mean(dim="timebounds")
# ).sel(scenario='ssp245', timebounds=np.arange(1850, 2102))
temperature_baseline_1850_1900 = (
    f.temperature.sel(layer=0) - f.temperature.sel(layer=0).weighted(weights).mean(dim="timebounds")
)

# %%
weights = np.zeros((752, n_scen, 841))
weights[254, :, :] = 0.5
weights[254:274, :, :] = 1
weights[274, :, :] = 0.5
weights = xr.DataArray(
    weights, 
    dims=f.temperature.sel(layer=0).dims, 
    coords=f.temperature.sel(layer=0).coords
)
# output[..., ivolc] = (
#     f.temperature.sel(layer=0) - f.temperature.sel(layer=0).weighted(weights).mean(dim="timebounds")
# ).sel(scenario='ssp245', timebounds=np.arange(1850, 2102))
temperature_baseline_2004_2023 = (
    f.temperature.sel(layer=0) - f.temperature.sel(layer=0).weighted(weights).mean(dim="timebounds")
) + 1.05

# %%
temperature_baseline_2004_2023

mod_scens = future.loc[0::55, "model":"scenario"].values
mod_scens

# %%
scen_mods = {mod_scen[1]:mod_scen[0] for mod_scen in mod_scens}

# %%
scen_mods

# %% [markdown]
# Fill in the data 
#
# The next few cells get repeated with the variables of interest



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
    temp_out_data = np.ones((752, 841*n_scen))*np.nan
    irow = 0
    for scenario in scenarios:
        # for config in valid_all:
        print(data_array.sel(scenario=scenario, config=valid_all))
        print(data_array.sel(scenario=scenario, config=valid_all).shape)
        temp_out_data[:, irow:irow+841] = data_array.sel(scenario=scenario, config=valid_all, timebounds=np.arange(startyear, endyear +1))
        irow = irow + 841
    if plot:
        plot_per_scenario(variable_name, unit, data_array)
    return pd.DataFrame(temp_out_data.T, index=index, columns=np.arange(startyear, endyear+1))

def plot_per_scenario(variable_name, unit, data_array):
    fig, axs = pl.subplots(1, 1, figsize=(10, 5))
    for scenario in scenarios:
        axs.plot(np.arange(startyear, endyear+1), data_array.sel(scenario=scenario, config=valid_all).median(dim="config"), label=scenario)
        axs.fill_between(
            np.arange(startyear, endyear+1),
            data_array.sel(scenario=scenario, config=valid_all).quantile(0.05, dim="config"),
            data_array.sel(scenario=scenario, config=valid_all).quantile(0.95, dim="config"),
            alpha=0.2
        )
    axs.set_title(f"{variable_name} ({unit})")
    axs.set_xlabel("Time (years)")
    axs.set_ylabel(f"{variable_name} ({unit})")
    axs.legend()
    fig.tight_layout()
    fig.savefig(f"../output/{variable_name.replace('|', '_').replace(' ', '_').replace("(GSAT)", "GSAT")}_{run_choice}.png", dpi=300)
    
temp_out = produce_variable_dataframe('Climate Assessment|Surface Temperature (GSAT)', 'K', temperature_baseline_2004_2023)
forcing_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing', 'W/m2', f.forcing_sum)
toa_out = produce_variable_dataframe('Climate Assessment|Top of Atmosphere Energy Imbalance', 'W/m2', f.toa_imbalance)

# %%
greenhouse_gases = [
    'CO2',
    'CH4',
    'N2O',
    'Sulfur',
    'BC',
    'OC',
    'NH3',
    'NOx',
    'VOC',
    'CO',
    'CFC-11',
    'CFC-12',
    'CFC-113',
    'CFC-114',
    'CFC-115',
    'HCFC-22',
    'HCFC-141b',
    'HCFC-142b',
    'CCl4',
    'CHCl3',
    'CH2Cl2',
    'CH3Cl',
    'CH3CCl3',
    'CH3Br',
    'Halon-1211',
    'Halon-1301',
    'Halon-2402',
    'CF4',
    'C2F6',
    'C3F8',
    'c-C4F8',
    'C4F10',
    'C5F12',
    'C6F14',
    'C7F16',
    'C8F18',
    'NF3',
    'SF6',
    'SO2F2',
    'HFC-125',
    'HFC-134a',
    'HFC-143a',
    'HFC-152a',
    'HFC-227ea',
    'HFC-23',
    'HFC-236fa',
    'HFC-245fa',
    'HFC-32',
    'HFC-365mfc',
    'HFC-4310mee',
]
forcing_ghg = f.forcing.sel(specie=greenhouse_gases).sum(dim='specie')
forcing_ghg_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing|Anthropogenic|Greenhouse Gases', 'W/m2', forcing_ghg)

# %%
aerosols = [
    'Aerosol-radiation interactions',
    'Aerosol-cloud interactions'
]
forcing_aerosols = f.forcing.sel(specie=aerosols).sum(dim='specie')
forcing_aerosols_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing|Anthropogenic|Aerosols', 'W/m2', forcing_aerosols)

# %%
natural = [
    'Solar',
    'Volcanic'
]

forcing_natural = f.forcing.sel(specie=natural).sum(dim='specie')
forcing_natural_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing|Natural', 'W/m2', forcing_natural)
forcing_solar = f.forcing.sel(specie='Solar')
forcing_solar_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing|Natural|Solar', 'W/m2', forcing_solar)
forcing_volcanic = f.forcing.sel(specie='Volcanic')
forcing_volcanic_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing|Natural|Volcanic', 'W/m2', forcing_volcanic)  

conc_co2 = f.concentration.sel(specie='CO2')
concentration_co2_out = produce_variable_dataframe('Climate Assessment|Concentration|CO2', 'ppm', conc_co2)

forcing_ozone = f.forcing.sel(specie='Ozone')
forcing_ozone_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing|Anthropogenic|Ozone', 'W/m2', forcing_ozone)

forcing_other = f.forcing_sum - forcing_ghg - forcing_aerosols - forcing_natural - forcing_ozone
forcing_other_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing|Anthropogenic|Other', 'W/m2', forcing_other)

forcing_ozone_strat = (f.concentration.sel(specie='Equivalent effective stratospheric chlorine') 
                       - f.species_configs["baseline_concentration"].sel(specie="Equivalent effective stratospheric chlorine")
                       )*f.species_configs["ozone_radiative_efficiency"].sel(specie="Equivalent effective stratospheric chlorine")

forcing_ozonestratospheric_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing|Anthropogenic|Ozone|Stratospheric', 'W/m2', forcing_ozone_strat)
data_out = pd.concat(
    (
        temp_out, 
        toa_out,
        forcing_out, 
        forcing_ghg_out, 
        forcing_aerosols_out, 
        forcing_natural_out, 
        forcing_solar_out, 
        forcing_volcanic_out, 
        forcing_ozone_out,
        forcing_other_out,
        forcing_ozonestratospheric_out,
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
data_out.to_csv(f'../output/climate_assessment_{run_choice}_{ending}.csv')