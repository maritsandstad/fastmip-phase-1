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

# %% [markdown]
# # Make nice input files for fair
#
# ScenarioMIP scenarios 1750-2500, email from Marit Sandstad 21.03.2026

# %%
import os, sys

import pandas as pd

# %%
#future_emissions_input = '../data/emissions/continuous_emissions_timeseries_1750_2500.csv'
future_emissions_input = '../data/emissions/emissions_1750-2500_vlcf-ctap_test.csv'
future_emissions_output = '../output/future_emissions_vl_cf_ctap_cleaned.csv'
#future_emissions_input = '../data/emissions/emissions_1750-2500_WIEMIP_v1.6.0_4.csv'
#future_emissions_output = '../output/future_emissions_wiemip_cleaned.csv'
future = pd.read_csv(future_emissions_input)

# %%
print(future.shape)
print(future.head())


# %%
future_species = list(future[future.variable.str.startswith('Emissions|')].variable.unique())
future_species = list(future.variable.unique())

# %%
print(future_species)

# %%
if 'CO2|Gross Positive Emissions' in future_species:
    future_species.remove('CO2|Gross Positive Emissions')
if 'CO2|Gross Removals' in future_species:
    future_species.remove('CO2|Gross Removals')

# %%
rename_map = {specie: specie.replace('Emissions|', '') for specie in future_species}

# %%
rename_map

# %%
rename_map['Emissions|CO2|AFOLU'] = 'CO2 AFOLU'
rename_map['Emissions|CO2|Energy and Industrial Processes'] = 'CO2 FFI'
rename_map['Emissions|cC4F8'] = 'c-C4F8'
rename_map['Emissions|HFC|HFC43-10'] = 'HFC-4310mee'

# %%
#rename_map

# %%
future_emissions = future

# %%
future_emissions_renamed = future_emissions.replace(rename_map)

# %%
future_emissions_renamed

# %%
#future_emissions_renamed = future_emissions_renamed.drop(columns=[str(year) for year in range(2000, 2023)])

# %%
units = list(future_emissions_renamed.unit.unique())

# %%
units

# %%
units_map = {unit: unit for unit in units}
col_rename_map = {}
for column in future_emissions_renamed.columns:
    if len(column) == 6 and column.endswith('.5'):
        col_rename_map[column] = column.replace('.5', '.0')
print(future_emissions_renamed.shape)
future_emissions_renamed = future_emissions_renamed.rename(columns=col_rename_map)
print(future_emissions_renamed.head())
# %%
units_map['Mt CO2/yr'] = 'Gt CO2/yr'
#units_map['MtCO2 / yr'] = 'Gt CO2/yr'
units_map['kt N2O/yr'] = 'Mt N2O/yr'

# %%
future_emissions_renamed_reunited = future_emissions_renamed.replace(units_map)

# %%
print(future_emissions_renamed_reunited.head())

# %%
future_emissions_renamed_reunited_dedafted = future_emissions_renamed_reunited.copy()

for var_dedaft in ['CO2 FFI', 'CO2 AFOLU', 'N2O']:
    future_emissions_renamed_reunited_dedafted.loc[future_emissions_renamed_reunited.variable == var_dedaft, '1750.0':] = (
        future_emissions_renamed_reunited.loc[future_emissions_renamed_reunited.variable == var_dedaft, '1750.0':]
    ) / 1000

# %%
future_emissions_renamed_reunited_dedafted.loc[future_emissions_renamed_reunited_dedafted.variable == 'CO2 FFI']

# %%
# # check alignment in 2023
# for specie in historical.variable.unique():
#     print(historical.loc[historical.variable==specie, '2023'].values[0] - future_emissions_renamed_reunited_dedafted.loc[future_emissions_renamed_reunited_dedafted.variable==specie, '2023'].values[0])

# %%
os.makedirs('../output', exist_ok=True)

# %%
future_emissions_renamed_reunited_dedafted.to_csv(future_emissions_output, index=False)

# %%
