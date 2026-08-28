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
import os, sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import copy

import fair
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

# %%
print(fair.__version__)

RCMIP3_VERSION = "v2.0.1"
FAIR_CALIBRATION = "1.6.0"

# Here we need to rerun one normal scenario without natural forcings
# To get the correct temperature baseline for the idalised experiments

# %%
# common datasets
master_concentrations = pd.read_csv(f"../data/RCMIP/rcmip_phase3_concentrations_{RCMIP3_VERSION}.csv")
master_emissions = pd.read_csv(f"../data/RCMIP/rcmip_phase3_emissions_{RCMIP3_VERSION}.csv")
master_forcing = pd.read_csv(f"../data/RCMIP/rcmip_phase3_forcing_{RCMIP3_VERSION}.csv") 

print(master_forcing["Scenario"].unique())

# %%
scenarios_full = ['ssp126', 'ssp119', 'ssp245', 'ssp370', 'ssp585', 'ssp434', 'ssp460', 'ssp534-over']

# %%
n_scen = len(scenarios_full)
run_w_variability = False


# %%
# variable mapping
variables_short = [var.split("|")[-1] for var in master_emissions.Variable.unique()] + [var.split("|")[-1] for var in master_forcing.Variable.unique()]
temp_dict = {var: var for var in variables_short}
for var in temp_dict:
    if var[:3]=='HFC':
        temp_dict[var] = f"{var[:3]}-{var[3:]}"
    elif var[:3]=='CFC':
        temp_dict[var] = f"{var[:3]}-{var[3:]}"
    elif var[:4]=='HCFC':
        temp_dict[var] = f"{var[:4]}-{var[4:]}"
    elif var[:5]=='Halon':
        temp_dict[var] = f"{var[:5]}-{var[5:]}"

temp_dict['cC4F8'] = "c-C4F8"
temp_dict["Energy and Industrial Processes"] = "CO2 FFI"
temp_dict["AFOLU"] = "CO2 AFOLU"
temp_dict["Land use"] = "Land Use"
temp_dict["Land use"] = "Land use"
RCMIP3_LOOKUP = {value: key for key, value in temp_dict.items()}
RCMIP3_LOOKUP["Albedo Change"] = "Land Use"

# %%
# unit dedafter
DEDAFTER = {specie: 1 for specie in RCMIP3_LOOKUP}
DEDAFTER["CO2 FFI"] = 0.001
DEDAFTER["CO2 AFOLU"] = 0.001
DEDAFTER["CO2"] = 0.001
DEDAFTER["N2O"] = 0.001
startyear = 1750
endyear = 2500
total_years = endyear - startyear + 1
# %%


species, properties = read_properties(
    "../data/fair_parameters_1.6.0/"
    "species_configs_properties.csv",
)

print("Defined the species and properties, now reading in the calibrated parameters and defining the configs")
df_configs = pd.read_csv(
    "../data/fair_parameters_1.6.0/"
    "calibrated_constrained_parameters.csv",
    index_col=0,
)

valid_all = df_configs.index
print(valid_all.shape)

print(df_configs.columns)
#print(df_configs['forcing_scale[Land Use]'])
#sys.exit(4)

def set_fair_species_properties_one_specie(f, specie, scenario):
    exp_emis = scenario
    exp_conc = scenario
    exp_forc = scenario
    if properties[specie]["input_mode"]=="concentration":
        working_concentrations = copy.deepcopy(
            master_concentrations.loc[
                (master_concentrations["Scenario"]==exp_conc) & (master_concentrations["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
                str(startyear):str(endyear-1)
            ].T
        )
        working_concentrations.index = pd.to_numeric(working_concentrations.index)
        working_concentrations.loc[startyear-1] = working_concentrations.loc[startyear]
        working_concentrations.loc[endyear] = 2 * working_concentrations.loc[endyear-1] - working_concentrations.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_concentrations.loc[midyear] = np.nan
        working_concentrations = working_concentrations.sort_index()
        working_concentrations = working_concentrations.interpolate()
        f.concentration.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_concentrations.loc[np.arange(startyear-0.5, endyear, 1)].values
    elif properties[specie]["input_mode"]=="emissions":
        f.emissions.loc[
            dict(
                timepoints=np.arange(startyear+0.5, endyear), 
                scenario=scenario,
                specie=specie
            )
        ] = master_emissions.loc[
            (master_emissions["Scenario"]==exp_emis) & (master_emissions["Variable"].str.endswith(f"|{RCMIP3_LOOKUP[specie]}")),
            str(startyear):str(endyear-1)
        ].T * DEDAFTER[specie]
    elif properties[specie]["input_mode"]=="forcing":
        if specie == "Irrigation":
            print(f.forcing.loc[
                dict(
                    timebounds=np.arange(startyear, endyear+1), 
                    scenario=scenario,
                    specie=specie
                )
            ].shape)
            f.forcing.loc[
                dict(
                    timebounds=np.arange(startyear, endyear+1), 
                    scenario=scenario,
                    specie=specie
                )
            ] = np.zeros((endyear-startyear+1, 1))
            return

        if specie == "Contrails and Contrail-induced Cirrus":
            forcing_scale = np.ones(len(f.configs))
        else:
            forcing_scale = df_configs[f"forcing_scale[{RCMIP3_LOOKUP[specie]}]"].values.squeeze()
        if specie == "Land use":
            specie_search = "Albedo Change"
        else:
            specie_search = specie
        working_forcing = copy.deepcopy(
            master_forcing.loc[
                (master_forcing["Scenario"]==exp_forc) & (master_forcing["Variable"].str.endswith(f"|{specie_search}")),
                str(startyear):str(endyear)
            ].T
        )
        working_forcing.index = pd.to_numeric(working_forcing.index)
        print(working_forcing)
        print(specie_search)
        #print(master_forcing.loc[(master_forcing["Scenario"]==exp_forc)]["Variable"].unique())
        working_forcing.loc[startyear-1] = working_forcing.loc[startyear]
        working_forcing.loc[endyear] = 2 * working_forcing.loc[endyear-1] - working_forcing.loc[endyear-2]
        for midyear in np.arange(startyear-0.5, endyear):
            working_forcing.loc[midyear] = np.nan
        working_forcing = working_forcing.sort_index()
        working_forcing = working_forcing.interpolate()
        f.forcing.loc[
            dict(
                timebounds=np.arange(startyear, endyear+1), 
                scenario=scenario,
                specie=specie
            )
        ] = working_forcing.loc[np.arange(startyear-0.5, endyear, 1)].values * forcing_scale

baseline_subtract = np.loadtxt('../output/baseline_subtract.txt')

def produce_variable_dataframe(variable_name, unit, data_array, plot = True, scenarioname = None):
    mi = []
    for config in valid_all:
        ix = (
            "idealised", 
            scenarioname, 
            'World', 
            variable_name,
            unit,
            config,
            'fair-2.2.4',
            '1.6.0-full',
        )
        mi.append(ix)
    index = pd.MultiIndex.from_tuples(mi, names=['model', 'scenario', 'region', 'variable', 'unit', 'ensemble_member', 'climate_model', 'calibration'])
    temp_out_data = np.ones((total_years, 841))*np.nan
    irow = 0
    print(data_array.shape)
    print(data_array.sel())
    temp_out_data[:, irow:irow+841] = data_array.sel(scenario=scenarioname, config=valid_all, timebounds=np.arange(startyear, endyear+1)).values
    irow = irow + 841
    if plot:
        plot_per_scenario(variable_name, unit, data_array, scenarioname)
    return pd.DataFrame(temp_out_data.T, index=index, columns=np.arange(0, total_years))

def plot_per_scenario(variable_name, unit, data_array, scenarioname):
    fig, axs = pl.subplots(1, 1, figsize=(10, 5))
    axs.plot(data_array.sel(scenario=scenario, config=valid_all).median(dim="config"), label=scenario)
    axs.fill_between(
        np.arange(0, total_years),
        data_array.sel(scenario=scenarioname, config=valid_all).quantile(0.05, dim="config"),
        data_array.sel(scenario=scenarioname, config=valid_all).quantile(0.95, dim="config"),
        alpha=0.2
    )
    axs.set_title(f"{variable_name} ({unit})")
    axs.set_xlabel("Time (years)")
    axs.set_ylabel(f"{variable_name} ({unit})")
    axs.legend()
    fig.tight_layout()
    fig.savefig(f"../output/{scenarioname}_{variable_name.replace('|', '_').replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=300)
    
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

def produce_full_df_per_scenario(f, temperature_baseline_2004_2023, scenarioname = None):
    temp_out = produce_variable_dataframe('Climate Assessment|Surface Temperature (GSAT)', 'K', temperature_baseline_2004_2023, scenarioname=scenarioname)
    forcing_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing', 'W/m2', f.forcing_sum, scenarioname=scenarioname)
    toa_out = produce_variable_dataframe('Climate Assessment|Top of Atmosphere Energy Imbalance', 'W/m2', f.toa_imbalance, scenarioname=scenarioname)

    greenhouse_gases = [
        'CO2',
        'CH4',
        'N2O',
    ]
    forcing_ghg = f.forcing.sel(specie=greenhouse_gases).sum(dim='specie')
    forcing_ghg_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing|Anthropogenic|Greenhouse Gases', 'W/m2', forcing_ghg, scenarioname=scenarioname)

    conc_co2 = f.concentration.sel(specie='CO2')
    concentration_co2_out = produce_variable_dataframe('Climate Assessment|Concentration|CO2', 'ppm', conc_co2, scenarioname=scenarioname)
    forcing_other_out = forcing_out - forcing_ghg_out 
    forcing_other = f.forcing_sum - forcing_ghg
    forc_other_out = produce_variable_dataframe('Climate Assessment|Effective Radiative Forcing|Anthropogenic|Other', 'W/m2', forcing_other, scenarioname=scenarioname)

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
    
    return data_out

def scenario_df_to_csv(data_out, scenario):

    os.makedirs('../output', exist_ok=True)
    if run_w_variability:
        ending = "full"
    else:
        ending = "forced"
    data_out.to_csv(f'../output/climate_assessment_cmip6_{scenario}_{ending}.csv')

for scenario in scenarios_full:
    print("Setting up FAIR for scenario {}".format(scenario))
    f = FAIR(ch4_method="Thornhill2021")
    scenarios = [scenario]
    f.define_time(startyear, endyear, 1)
    f.define_scenarios(scenarios)
    f.define_species(species, properties)
    print("Read in the calibrated parameters and defined the configs, now allocating memory for the model runs")
    f.define_configs(valid_all)
    f.allocate()
    print("FaIR is set up, now filling in the emissions and forcing data")

    for specie in species:
        set_fair_species_properties_one_specie(f, specie, scenario)
    f.fill_species_configs(
        "../data/fair_parameters_1.6.0/species_configs_properties.csv",
    )
    f.override_defaults(
    "../data/fair_parameters_1.6.0/calibrated_constrained_parameters.csv",
    )
    if run_w_variability:
        fill(f.climate_configs['stochastic_run'], True)
    else:
        fill(f.climate_configs['stochastic_run'], False)
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    print(f.species_configs['forcing_reference_concentration'])
    f.run()
    temperature_baseline_2004_2023 = (
        f.temperature.sel(layer=0) - baseline_subtract[0,:]
    ) + 1.05

    # %%
    temperature_baseline_2004_2023
    data_out = produce_full_df_per_scenario(f, temperature_baseline_2004_2023, scenarioname=scenario)
    scenario_df_to_csv(data_out, scenario)