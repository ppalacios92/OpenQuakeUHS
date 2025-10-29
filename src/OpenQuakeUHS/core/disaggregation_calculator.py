import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import os
import geopandas as gpd

class Disaggregation:

    def __init__(self, shp_path_ecuador , base_path , target_poe , target_imt , PRY ,save_path=None  ):
  
        self.shp_path_ecuador=shp_path_ecuador
        self.base_path = base_path
        self.target_poe = target_poe
        self.target_imt = target_imt
        self.save_path=save_path
        self.PRY=PRY

        self.csv_files()
        self.plot_disaggregation()

    def csv_files(self):
        
        # === Buscar archivos CSV ===
        csv_files = [f for f in os.listdir(self.base_path) if f.endswith('.csv')]

        # === Buscar archivos específicos ===
        file_data = next((f for f in csv_files if "Mag_Dist_Eps-mean" in f), None)
        file_data_TRT = next((f for f in csv_files if "TRT_Mag_Dist-mean" in f), None)
        file_lon_lat = next((f for f in csv_files if f.startswith('TRT_Lon_Lat-mean')), None)

        # === Verificación simple ===
        if file_data is None or file_data_TRT is None:
            raise FileNotFoundError("No se encontraron los archivos esperados en la carpeta.")
        if file_lon_lat is None:
            raise FileNotFoundError("No se encontró el archivo 'TRT_Lon_Lat-mean*.csv' en la carpeta.")

        # === Cargar archivos ===
        self.data = pd.read_csv(os.path.join(self.base_path, file_data), comment='#')
        self.data_TRT = pd.read_csv(os.path.join(self.base_path, file_data_TRT), comment='#')
        self.data_lon_lat = pd.read_csv(os.path.join(self.base_path, file_lon_lat), comment='#')


    
        return self.data , self.data_TRT, self.data_lon_lat 
    

    def disaggregation_mod_mean(self):
        data=self.data
        # === Filtrar por poe ===
        data01 = data[(data['poe'] == self.target_poe) & (data['imt'] == self.target_imt)].copy()

        # Buscar la columna que comience con 'mean' o 'rlz'
        hz_key = [col for col in data01.columns if col.startswith('mean') or col.startswith('rlz')][0]
        # Normalizar
        data01['hz_cont'] = data01[hz_key] / data01[hz_key].sum()
        # === Calcular mod y media ===
        data_reduced = data01.groupby(['mag', 'dist'])['hz_cont'].sum().reset_index()
        modal_row = data_reduced.sort_values(by='hz_cont', ascending=False).iloc[0]
        mod_mag = modal_row['mag']
        mod_dist = modal_row['dist']
        mean_mag = np.sum(data_reduced['mag'] * data_reduced['hz_cont'])
        mean_dist = np.sum(data_reduced['dist'] * data_reduced['hz_cont'])

        # === Pivotear hz_cont para obtener columnas por epsilon ===
        d01 = data01.pivot_table(values='hz_cont', index=['mag','dist'], columns='eps', aggfunc=np.sum, fill_value=0).reset_index()

        # === Obtener lista de epsilons ===
        eps_vals = [v for v in d01.columns[2:]]
        cmap = cm.get_cmap('jet', len(eps_vals))

        clrs = [cmap(i) for i in range(len(eps_vals))]  

        # === Renombrar columnas de eps para facilidad ===
        d01.rename(columns=dict(zip(eps_vals, ['ep'+str(i) for i in range(len(eps_vals))])), inplace=True)

        # === Asignar colores por columna ===
        for j in range(len(eps_vals)):
            d01['c'+str(j)] = [clrs[j]] * len(d01)

        # === Calcular alturas acumuladas para apilamiento vertical ===
        d01['ez0'] = np.zeros(len(d01))
        for j in range(1, len(eps_vals)):
            d01['ez'+str(j)] = d01['ez'+str(j-1)] + d01['ep'+str(j-1)]

        # === Reformatear a arreglo largo: dist, mag, dz, base_z, color ===
        dat = np.vstack([
            d01[['dist', 'mag', f'ep{i}', f'ez{i}', f'c{i}']].values
            for i in range(len(eps_vals))
        ])

        return dat , mod_mag , mod_dist , mean_mag , mean_dist, clrs, eps_vals


    def disaggregation_TRT(self):
        data_TRT=self.data_TRT
        data_TRT =  data_TRT[(data_TRT['poe'] == self.target_poe) & (data_TRT['imt'] == self.target_imt)].copy()
        # === Calcular contribución porcentual ===
        data_TRT['hz_cont_TRT'] = data_TRT['mean'] / data_TRT['mean'].sum()
        # === Codificar por 'trt' con colores 'tab:' ===
        trt_unique_TRT = data_TRT['trt'].unique()
        tab_colors_TRT = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        trt_to_color_TRT = {trt: tab_colors_TRT[i % len(tab_colors_TRT)] for i, trt in enumerate(trt_unique_TRT)}
        data_TRT['color_TRT'] = data_TRT['trt'].map(trt_to_color_TRT)

        return data_TRT, trt_to_color_TRT , trt_unique_TRT

    def disaggregation_lon_lat(self):  

        # === Leer datos sísmicos por lon/lat ===
        data_lon_lat =self.data_lon_lat
        # === Filtrar por IMT y PoE objetivo ===
        data_lon_lat_filt = data_lon_lat[
            (data_lon_lat['imt'] == self.target_imt) & 
            (data_lon_lat['poe'] == self.target_poe)
        ].copy()
        # === Calcular contribución porcentual ===
        data_lon_lat_filt['hz_cont_lon_lat'] = (
            data_lon_lat_filt['mean'] / data_lon_lat_filt['mean'].sum()
        )
        # === Clasificar por tipo tectónico (trt) ===
        trt_unique_lon_lat = data_lon_lat_filt['trt'].unique()
        # === Asignar colores únicos por trt (paleta tab:) ===
        tab_colors_TRT = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        trt_to_color_TRT = {
            trt: tab_colors_TRT[i % len(tab_colors_TRT)] 
            for i, trt in enumerate(trt_unique_lon_lat)
        }
        # === Asignar color a cada fila
        data_lon_lat_filt['color_TRT'] = data_lon_lat_filt['trt'].map(trt_to_color_TRT)

        return data_lon_lat_filt

    def read_shp_map(self):
        # === Leer y reproyectar shapefile a EPSG:4326 ===
        gdf_ecuador = gpd.read_file(self.shp_path_ecuador)
        gdf_ecuador = gdf_ecuador.to_crs(epsg=4326)
        return gdf_ecuador


    
    def plot_disaggregation(self):
        data , mod_mag , mod_dist , mean_mag , mean_dist, clrs, eps_vals= self.disaggregation_mod_mean()
        data_TRT, trt_to_color_TRT, trt_unique_TRT = self.disaggregation_TRT()
        data_lon_lat_filt=self.disaggregation_lon_lat() 
        gdf_ecuador=self.read_shp_map()
        


        # === Crear figura con 3 subplots ===
        fig = plt.figure(figsize=(18, 6))

        # === Subplot 1: Disaggregation por epsilon ===
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.set_ylabel("Mw", fontweight='bold')
        ax1.set_xlabel("Distance (km)", fontweight='bold')
        ax1.set_zlabel("Hazard Contribution (%)", fontweight='bold')

        ax1.bar3d(
            x=data[:, 0].astype(float),  
            y=data[:, 1].astype(float),  
            z=data[:, 3].astype(float) * 100,  
            dx=15.0,
            dy=0.15,
            dz=data[:, 2].astype(float) * 100, 
            shade=True,
            color=data[:, 4],
            zsort='max',
            alpha=0.5,
        )

        ax1.set_title(
            f'Disaggregation Plot by Epsilon\n'
            # f'Mod: Mw={mod_mag:.2f}, R={mod_dist:.0f} km | '
            f'Mean: Mw={mean_mag:.2f}, R={mean_dist:.0f} km',
            fontweight='bold'
        )

        # === Subplot 2: Disaggregation por TRT ===
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.set_ylabel("Mw", fontweight='bold')
        ax2.set_xlabel("Distance (km)", fontweight='bold')
        ax2.set_zlabel("Hazard Contribution (%)", fontweight='bold')

        ax2.bar3d(
            x=data_TRT['dist'].astype(float),
            y=data_TRT['mag'].astype(float),
            z=0.0,
            dx=15.0,
            dy=0.15,
            dz=data_TRT['hz_cont_TRT'] * 100,
            color=data_TRT['color_TRT'],
            shade=True,
            zsort='max',
            alpha=0.8,
        )

        ax2.set_title(
            f"Disaggregation Plot by TRT\n"
            f"(poe={self.target_poe}, imt={self.target_imt})",
            fontweight='bold'
        )

        # === Subplot 3: Contribución por Lon/Lat + shapefile ===
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')

        dx = 0.2
        dy = 0.2
        dz = data_lon_lat_filt['hz_cont_lon_lat'] * 100  # en porcentaje

        ax3.bar3d(
            x=data_lon_lat_filt['lon'],
            y=data_lon_lat_filt['lat'],
            z=np.zeros_like(dz),
            dx=dx,
            dy=dy,
            dz=dz,
            color=data_lon_lat_filt['color_TRT'],
            shade=True,
            alpha=0.8
        )

        # === Dibujar shapefile reproyectado en z=0 ===
        for geom in gdf_ecuador.geometry:
            if geom.geom_type == 'Polygon':
                x, y = geom.exterior.xy
                ax3.plot(x, y, zs=0, zdir='z', color='black', linewidth=1)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    ax3.plot(x, y, zs=0, zdir='z', color='black', linewidth=1)

        # === Etiquetas ===
        ax3.set_xlabel("Longitude", fontweight='bold')
        ax3.set_ylabel("Latitude", fontweight='bold')
        ax3.set_zlabel("Hazard Contribution (%)", fontweight='bold')

        ax3.set_title(
            f"Hazard Contribution by Location",
            fontweight='bold'
        )

        # === Leyendas ===
        handles_eps = [mpatches.Patch(color=clrs[i], label=str(eps_vals[i])) for i in range(len(eps_vals))]
        handles_TRT = [mpatches.Patch(color=trt_to_color_TRT[trt], label=trt) for trt in trt_unique_TRT]
        handle_hz_cont = [mpatches.Patch(color='royalblue', label='Hazard Contribution')]

        # Leyenda 1: Epsilon
        leg1 = fig.legend(handles=handles_eps, title='Epsilon',
                        loc='lower center', bbox_to_anchor=(0.25, -0.05),
                        ncol=int(len(eps_vals)/2), frameon=False)
        leg1.get_title().set_fontweight('bold')

        # Leyenda 2: Tectonic Region Type
        leg2 = fig.legend(handles=handles_TRT, title='Tectonic Region Type',
                        loc='lower center', bbox_to_anchor=(0.63, -0.01),
                        ncol=3, frameon=False)
        leg2.get_title().set_fontweight('bold')

        # # Leyenda 3: Contribución (sin título)
        # fig.legend(handles=handle_hz_cont, title='',
        #         loc='lower center', bbox_to_anchor=(0.8, -0.01),
        #         frameon=False)


        fig.text(
            0.99, -0.01,
            f"PRY: {self.PRY}\n© 2025 - Patricio Palacios B.",
            ha='right', va='top', fontsize=9, color='gray', style='italic', multialignment='right'
        )


        if self.save_path:
            fig.savefig(f"{self.save_path}_disaggregation.svg", format="svg", bbox_inches="tight")
            fig.savefig(f"{self.save_path}_disaggregation.pdf", format="pdf", bbox_inches="tight")
            print(f"Figures saved to {self.save_path}_disaggregation.(svg/pdf)")

        else:
            plt.show()
