import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def nwq(turnout, partyPercent, partyName):
    # turnout:      voter turnout in percent
    # partyPercent: percent turnout for the party
    # partyName:    name of the party
    
    plt.scatter(turnout, partyPercent)
    plt.ylabel('Anteil Zweitstimme in %')
    plt.xlabel('Wahlbeteiligung in %')
    plt.title(partyName)
    plt.show()

# load *.csv file
# get kerg.cv from: https://www.bundeswahlleiter.de/bundestagswahlen/2021/ergebnisse/opendata/csv/
rawdata = pd.read_csv('kerg.csv', encoding='UTF-8', skiprows=2, dtype=str, sep=';')

# remove empty rows
data = rawdata.dropna(subset=['gehört zu'])

# remove non-Wahlkreis rows
data = data[data['gehört zu'].map(int) <= 16]

waehlende =  data['Wählende'].to_numpy().astype(float) 
wahlberechtigte = data['Wahlberechtigte'].to_numpy().astype(float)

wahlbeteiligung = 100 * waehlende / wahlberechtigte

print('Höchste Wahlbeteiligung: {:.1f} in WK: {} ({})'.format(max(wahlbeteiligung), data.iloc[np.argmax(wahlbeteiligung), 0], data.iloc[np.argmax(wahlbeteiligung), 1]))
print('Niedrigste Wahlbeteiligung: {:.1f} in WK: {} ({})'.format(min(wahlbeteiligung), data.iloc[np.argmin(wahlbeteiligung), 0], data.iloc[np.argmin(wahlbeteiligung), 1]))

wahlkreis = data['Nr'].to_numpy().astype(int)

# create dataFrame
wb = np.vstack((wahlkreis, wahlbeteiligung)).T
wb_df = pd.DataFrame(data=wb, columns=['WKR_NR', 'Wahlbeteiligung'])


## now get the shapefile data
map_df = gpd.read_file('generalizedShape/Geometrie_Wahlkreise_20DBT_geo.shx')

# merge with election data
df_merged = map_df.merge(wb_df, left_on=['WKR_NR'], right_on=['WKR_NR'])


fig = plt.figure(figsize=(6, 8), dpi=150)
#ax = plt.subplot2grid((1, 10), (0, 0), colspan=7)
ax = fig.add_axes([0.025, 0.025, 0.75, 0.9])

cmap = mpl.cm.plasma
cmaplist = [cmap(i) for i in range(cmap.N)]
# new colormap
cmap = mpl.colors.LinearSegmentedColormap.from_list('CustomBlues', cmaplist, cmap.N)
#bounds = np.arange(62.5, 87.5, 2.5)
bounds = np.arange(63, 86, 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

df_merged.plot(column='Wahlbeteiligung', cmap=cmap, linewidth=0.25, ax=ax,
        edgecolor='0.0', legend=False, vmin=bounds[0], vmax=bounds[-1], norm=norm)
ax.axis('off')
ax.set_title('Wahlbeteiligung in %', fontsize=20)

ax2 = fig.add_axes([0.825, 0.075, 0.05, 0.775])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', 
        ticks=bounds[::2], boundaries=bounds, format='%d')
cb.ax.tick_params(labelsize=15)


plt.figtext(0.01, 0.01, 'Erst- und Zweitstimme bei der Bundestagswahl 2021\nVorläufiges Ergebnis (Stand: 28.09.2021)', ha='left', va='bottom', fontsize=10)

plt.show()
fig.savefig('wahlbeteiligung_2021.svg')
