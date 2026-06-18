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
import xarray as xr

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

# %%
# historical = pd.read_csv('../data/emissions/historical_emissions_1750-2023_cmip7.csv')

# %%
# historical.loc[historical.variable=='Halon-1202'].index

# %%
# historical = historical.drop(index=historical.loc[historical.variable=='Halon-1202'].index)

# %%
# emitted_species = list(historical.variable.values)

# %%
future = pd.read_csv('../output/future_emissions_cleaned.csv')

# %%
future.loc[future.variable=='Halon-1202'].index

# %%
future = future.drop(index=future.loc[future.variable=='Halon-1202'].index)

# %%
future = future.drop(index=future.loc[future.variable=='Emissions|CO2|Gross Positive Emissions'].index)

# %%
future = future.drop(index=future.loc[future.variable=='Emissions|CO2|Gross Removals'].index)

# %%
emitted_species = list(future.variable.unique())

# %%
scenarios = list(future.scenario.unique())

# %%
scenarios

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
scenarios_mapping = {
    #'SSP2 - Low Overshoot_a': 'LN',
    #'SSP3 - High Emissions': 'H',
    'SSP2 - Medium Emissions': 'M',
    #'SSP2 - Low Emissions': 'L',
    #'SSP1 - Very Low Emissions': 'VL',
    #'SSP5 - Medium-Low Emissions_a': 'HL',
    #'SSP2 - Medium-Low Emissions': 'ML'
}

# %%
volcanic_forcing = df_volcanic["volcanic_erf_rel_1850-2021"].loc[1750:2501].values
solar_forcing = df_solar["solar_erf_rel_1850-2019"].loc[1750:2501].values

print("Loaded in data, now ready to define the FaIR runs")

# %% [markdown]
# ## With internal variability

# %%
f = FAIR(ch4_method="Thornhill2021")
print("Set up the FaIR object, now defining the time and scenarios")
f.define_time(1750, 2501, 1)
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
        # f.emissions.loc[dict(timepoints=np.arange(1750.5, 2024), scenario=scenario, specie=specie)] = (
        #     historical.loc[historical.variable==specie, '1750':].T
        # )
        f.emissions.loc[dict(timepoints=np.arange(1750.5, 2501), scenario=scenario, specie=specie)] = (
            future.loc[(future.variable==specie) & (future.scenario==scenario), '1750.0':].T
        )
    f.forcing.loc[dict(scenario=scenario, specie='Land use')] = (
        df_landuse.loc[1750:2501, scenarios_mapping[scenario]].values[:, None] * df_configs["forcing_scale[Land use]"].values.squeeze()
    )
    f.forcing.loc[dict(scenario=scenario, specie='Irrigation')] = (
        df_irrigation.loc[1750:2501, scenarios_mapping[scenario]].values[:, None] * df_configs["forcing_scale[Irrigation]"].values.squeeze()
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

# %%
pl.plot(temperature_baseline_2004_2023.median(dim="config"))
pl.plot(temperature_baseline_1850_1900.median(dim="config"))

# %%
pl.plot(temperature_baseline_1850_1900.median(dim="config"))

# %%
pl.plot((
    f.temperature.sel(layer=0) -
    f.temperature.sel(timebounds = np.arange(1850, 1902), layer=0).mean(dim="timebounds")
).median(dim="config"))

# %%
pl.plot(f.forcing_sum.median(dim="config"))

# %%
pl.plot(f.temperature.sel(layer=0, config=valid_all[0]))

# %%
(
    temperature_baseline_1850_1900
).median(dim="config").max(dim="timebounds")

# %%
pl.plot(temperature_baseline_1850_1900.sel(scenario=scenarios[0]));

# %%
pl.plot(temperature_baseline_2004_2023.sel(scenario=scenarios[0]));

# %%
[f'Climate Assessment|Surface Temperature (GSAT)|ensemble member {config} [fair-2.2.4 cal-1.6.0]' for config in valid_all]

# %%
# index = pd.MultiIndex.from_product(
#     [
#         scenarios,
#         ['World'],
#         [f'Climate Assessment|Surface Temperature (GSAT)|ensemble member {config} [fair-2.2.4 cal-1.6.0]' for config in valid_all],
#         ['K']
#     ],
#     names=["scenario", "region", "variable", "unit"]
# )

# %%
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

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Surface Temperature (GSAT)',
            'K',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

# %%
index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])

# %%
temp_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    temp_out_data[:, irow:irow+841] = temperature_baseline_2004_2023.sel(scenario=scenario, config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841

# %%
temp_out_data

# %%
temp_out = pd.DataFrame(temp_out_data.T, index=index, columns=np.arange(1850, 2502))

# %%
temp_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

# %%
index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])

# %%
forcing_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_out_data[:, irow:irow+841] = f.forcing_sum.sel(scenario=scenario, config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841

# %%
forcing_out = pd.DataFrame(forcing_out_data.T, index=index, columns=np.arange(1850, 2502))

# %%
forcing_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Top of Atmosphere Energy Imbalance',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
toa_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    toa_out_data[:, irow:irow+841] = f.toa_imbalance.sel(scenario=scenario, config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841
toa_out = pd.DataFrame(toa_out_data.T, index=index, columns=np.arange(1850, 2502))
toa_out

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

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Anthropogenic|Greenhouse Gases',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_ghg_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_ghg_out_data[:, irow:irow+841] = f.forcing.sel(scenario=scenario, specie=greenhouse_gases, config=valid_all, timebounds=np.arange(1850, 2502)).sum(dim='specie')
    irow = irow + 841

# %%
forcing_ghg_out = pd.DataFrame(forcing_ghg_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_ghg_out

# %%
aerosols = [
    'Aerosol-radiation interactions',
    'Aerosol-cloud interactions'
]

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Anthropogenic|Aerosols',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_aerosols_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_aerosols_out_data[:, irow:irow+841] = f.forcing.sel(scenario=scenario, specie=aerosols, config=valid_all, timebounds=np.arange(1850, 2502)).sum(dim='specie')
    irow = irow + 841

forcing_aerosols_out = pd.DataFrame(forcing_aerosols_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_aerosols_out

# %%
natural = [
    'Solar',
    'Volcanic'
]

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Natural',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_natural_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_natural_out_data[:, irow:irow+841] = f.forcing.sel(scenario=scenario, specie=natural, config=valid_all, timebounds=np.arange(1850, 2502)).sum(dim='specie')
    irow = irow + 841

forcing_natural_out = pd.DataFrame(forcing_natural_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_natural_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Concentration|CO2',
            'ppm',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
concentration_co2_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    concentration_co2_out_data[:, irow:irow+841] = f.concentration.sel(scenario=scenario, specie='CO2', config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841

concentration_co2_out = pd.DataFrame(concentration_co2_out_data.T, index=index, columns=np.arange(1850, 2502))
concentration_co2_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Natural|Solar',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_solar_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_solar_out_data[:, irow:irow+841] = f.forcing.sel(scenario=scenario, specie="Solar", config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841

forcing_solar_out = pd.DataFrame(forcing_solar_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_solar_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Natural|Volcanic',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_volcanic_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_volcanic_out_data[:, irow:irow+841] = f.forcing.sel(scenario=scenario, specie="Volcanic", config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841

forcing_volcanic_out = pd.DataFrame(forcing_volcanic_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_volcanic_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Anthropogenic|Ozone',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_ozone_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_ozone_out_data[:, irow:irow+841] = f.forcing.sel(scenario=scenario, specie="Ozone", config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841

forcing_ozone_out = pd.DataFrame(forcing_ozone_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_ozone_out

# %%
forcing_out_data - forcing_natural_out_data - forcing_ozone_out_data - forcing_ghg_out_data - forcing_aerosols_out_data

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Anthropogenic|Other',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_other_out_data = forcing_out_data - forcing_natural_out_data - forcing_ozone_out_data - forcing_ghg_out_data - forcing_aerosols_out_data

forcing_other_out = pd.DataFrame(forcing_other_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_other_out

# %%
f.species_configs["ozone_radiative_efficiency"].sel(specie="Equivalent effective stratospheric chlorine")

# %%
f.species_configs["baseline_concentration"].sel(specie="Equivalent effective stratospheric chlorine")

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Anthropogenic|Ozone|Stratospheric',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_ozonestratospheric_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_ozonestratospheric_out_data[:, irow:irow+841] = (
        f.concentration.sel(scenario=scenario, specie='Equivalent effective stratospheric chlorine', config=valid_all, timebounds=np.arange(1850, 2502)) -
        f.species_configs["baseline_concentration"].sel(specie="Equivalent effective stratospheric chlorine")
    ) * f.species_configs["ozone_radiative_efficiency"].sel(specie="Equivalent effective stratospheric chlorine")
    irow = irow + 841

forcing_ozonestratospheric_out = pd.DataFrame(forcing_ozonestratospheric_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_ozonestratospheric_out

# %%
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
data_out.to_csv('../output/climate_assessment_full.csv')

# %% [markdown]
# ## Without internal variability

# %%
f = FAIR(ch4_method="Thornhill2021")

f.define_time(1750, 2501, 1)
f.define_scenarios(scenarios)

species, properties = read_properties(
    "../data/fair_parameters_1.6.0/"
    "species_configs_properties.csv",
)

f.define_species(species, properties)
df_configs = pd.read_csv(
    "../data/fair_parameters_1.6.0/"
    "calibrated_constrained_parameters.csv",
    index_col=0,
)

valid_all = df_configs.index

f.define_configs(valid_all)
f.allocate()

# %%
for scenario in scenarios:
    for specie in emitted_species:
        f.emissions.loc[dict(timepoints=np.arange(1750.5, 2501), scenario=scenario, specie=specie)] = (
            future.loc[(future.variable==specie) & (future.scenario==scenario), '1750.0':].T
        )
    f.forcing.loc[dict(scenario=scenario, specie='Land use')] = (
        df_landuse.loc[1750:2501, scenarios_mapping[scenario]].values[:, None] * df_configs["forcing_scale[Land use]"].values.squeeze()
    )
    f.forcing.loc[dict(scenario=scenario, specie='Irrigation')] = (
        df_irrigation.loc[1750:2501, scenarios_mapping[scenario]].values[:, None] * df_configs["forcing_scale[Irrigation]"].values.squeeze()
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
f.override_defaults(
    "../data/fair_parameters_1.6.0/calibrated_constrained_parameters.csv",
)

# %%
fill(f.climate_configs['stochastic_run'], False)# = np.zeros(841, dtype=bool)

# %%
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

# %%
f.run()

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

baseline_subtract = f.temperature.sel(layer=0).weighted(weights).mean(dim="timebounds")
print(baseline_subtract)
baseline_subtract.to_dataframe().to_csv('../output/baseline_subtract.csv')

# %%
temperature_baseline_2004_2023

# %%
pl.plot(temperature_baseline_2004_2023.median(dim="config"))
pl.plot(temperature_baseline_1850_1900.median(dim="config"))

# %%
pl.plot(temperature_baseline_1850_1900.median(dim="config"))

# %%
pl.plot((
    f.temperature.sel(layer=0) -
    f.temperature.sel(timebounds = np.arange(1850, 1902), layer=0).mean(dim="timebounds")
).median(dim="config"))

# %%
pl.plot(f.forcing_sum.median(dim="config"))

# %%
pl.plot(f.temperature.sel(layer=0, config=valid_all[0]))

# %%
(
    temperature_baseline_1850_1900
).median(dim="config").max(dim="timebounds")

# %%
pl.plot(temperature_baseline_1850_1900.sel(scenario=scenarios[0]));

# %%
pl.plot(temperature_baseline_2004_2023.sel(scenario=scenarios[0]));

# %%
[f'Climate Assessment|Surface Temperature (GSAT)|ensemble member {config} [fair-2.2.4 cal-1.6.0]' for config in valid_all]

# %%
# index = pd.MultiIndex.from_product(
#     [
#         scenarios,
#         ['World'],
#         [f'Climate Assessment|Surface Temperature (GSAT)|ensemble member {config} [fair-2.2.4 cal-1.6.0]' for config in valid_all],
#         ['K']
#     ],
#     names=["scenario", "region", "variable", "unit"]
# )

# %%
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

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Surface Temperature (GSAT)',
            'K',
            config,
            'fair-2.2.4',
            '1.6.0-forced',
        )
        mi.append(ix)

# %%
index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])

# %%
temp_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    temp_out_data[:, irow:irow+841] = temperature_baseline_2004_2023.sel(scenario=scenario, config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841

# %%
temp_out_data

# %%
temp_out = pd.DataFrame(temp_out_data.T, index=index, columns=np.arange(1850, 2502))

# %%
temp_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-forced',
        )
        mi.append(ix)

# %%
index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])

# %%
forcing_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_out_data[:, irow:irow+841] = f.forcing_sum.sel(scenario=scenario, config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841

# %%
forcing_out = pd.DataFrame(forcing_out_data.T, index=index, columns=np.arange(1850, 2502))

# %%
forcing_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Top of Atmosphere Energy Imbalance',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
toa_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    toa_out_data[:, irow:irow+841] = f.toa_imbalance.sel(scenario=scenario, config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841
toa_out = pd.DataFrame(toa_out_data.T, index=index, columns=np.arange(1850, 2502))
toa_out

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

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Anthropogenic|Greenhouse Gases',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-forced',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_ghg_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_ghg_out_data[:, irow:irow+841] = f.forcing.sel(scenario=scenario, specie=greenhouse_gases, config=valid_all, timebounds=np.arange(1850, 2502)).sum(dim='specie')
    irow = irow + 841

# %%
forcing_ghg_out = pd.DataFrame(forcing_ghg_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_ghg_out

# %%
aerosols = [
    'Aerosol-radiation interactions',
    'Aerosol-cloud interactions'
]

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Antropogenic|Aerosols',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-forced',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_aerosols_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_aerosols_out_data[:, irow:irow+841] = f.forcing.sel(scenario=scenario, specie=aerosols, config=valid_all, timebounds=np.arange(1850, 2502)).sum(dim='specie')
    irow = irow + 841

forcing_aerosols_out = pd.DataFrame(forcing_aerosols_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_aerosols_out

# %%
natural = [
    'Solar',
    'Volcanic'
]

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Natural',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-forced',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_natural_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_natural_out_data[:, irow:irow+841] = f.forcing.sel(scenario=scenario, specie=natural, config=valid_all, timebounds=np.arange(1850, 2502)).sum(dim='specie')
    irow = irow + 841

forcing_natural_out = pd.DataFrame(forcing_natural_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_natural_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Concentration|CO2',
            'ppm',
            config,
            'fair-2.2.4',
            '1.6.0-forced',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
concentration_co2_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    concentration_co2_out_data[:, irow:irow+841] = f.concentration.sel(scenario=scenario, specie='CO2', config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841

concentration_co2_out = pd.DataFrame(concentration_co2_out_data.T, index=index, columns=np.arange(1850, 2502))
concentration_co2_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Natural|Solar',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_solar_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_solar_out_data[:, irow:irow+841] = f.forcing.sel(scenario=scenario, specie="Solar", config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841

forcing_solar_out = pd.DataFrame(forcing_solar_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_solar_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Natural|Volcanic',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_volcanic_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_volcanic_out_data[:, irow:irow+841] = f.forcing.sel(scenario=scenario, specie="Volcanic", config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841

forcing_volcanic_out = pd.DataFrame(forcing_volcanic_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_volcanic_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Anthropogenic|Ozone',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_ozone_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_ozone_out_data[:, irow:irow+841] = f.forcing.sel(scenario=scenario, specie="Ozone", config=valid_all, timebounds=np.arange(1850, 2502))
    irow = irow + 841

forcing_ozone_out = pd.DataFrame(forcing_ozone_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_ozone_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Anthropogenic|Other',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_other_out_data = forcing_out_data - forcing_natural_out_data - forcing_ozone_out_data - forcing_ghg_out_data - forcing_aerosols_out_data

forcing_other_out = pd.DataFrame(forcing_other_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_other_out

# %%
mi = []
for scenario in scenarios:
    for config in valid_all:
        ix = (
            scen_mods[scenario], 
            scenario, 
            'World', 
            'Climate Assessment|Effective Radiative Forcing|Anthropogenic|Ozone|Stratospheric',
            'W/m2',
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)

index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
forcing_ozonestratospheric_out_data = np.ones((652, 841*n_scen))*np.nan
irow = 0
for scenario in scenarios:
    # for config in valid_all:
    forcing_ozonestratospheric_out_data[:, irow:irow+841] = (
        f.concentration.sel(scenario=scenario, specie='Equivalent effective stratospheric chlorine', config=valid_all, timebounds=np.arange(1850, 2502)) -
        f.species_configs["baseline_concentration"].sel(specie="Equivalent effective stratospheric chlorine")
    ) * f.species_configs["ozone_radiative_efficiency"].sel(specie="Equivalent effective stratospheric chlorine")
    irow = irow + 841

forcing_ozonestratospheric_out = pd.DataFrame(forcing_ozonestratospheric_out_data.T, index=index, columns=np.arange(1850, 2502))
forcing_ozonestratospheric_out

# %%
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
data_out.to_csv('../output/climate_assessment_forced.csv')

# %%
