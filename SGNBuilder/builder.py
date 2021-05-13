import os
from SGNBuilder import builder_utils as bu
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm, trange

class NetworkBuilder:
    def __init__(self, base_path, census_path):
        self.__BASE_PATH = base_path
        self.__CENSUS_PATH = census_path

    @property
    def base_path(self):
        return self.__BASE_PATH
    
    @property
    def census_path(self):
        return self.__CENSUS_PATH

    @base_path.setter
    def base_path(self, base_path):
        p = Path(base_path)
        self._validate_path(p)
        self.__BASE_PATH = p
    
    @census_path.setter
    def census_path(self, census_path):
        p = Path(census_path)
        self._validate_path(p)
        self.__CENSUS_PATH = p

    def _validate_path(self, path):
        if not os.path.exists(path):
            raise ValueError("Not a valid folder path")

    
    def _construct_folder_name(self, date):
        """
        Base directory is the path to the social distance metrics folder
        """
        year = str(date.year)
        month = str(date.month).zfill(2)
        day = str(date.day).zfill(2)
        return Path(self.__BASE_PATH, year, month, day)


    def _get_file_path(self, date_timestamp):
        filepath = self._construct_folder_name(date_timestamp)
        filename = os.listdir(filepath)[0]
        return Path(filepath,filename)


    def construct_filepaths_from_dates(self, start_date, end_date):
        analysis_dates = pd.date_range(start_date, end_date, freq='d')
        analysis_filepaths = [self._get_file_path(date) for date in analysis_dates]
        return analysis_filepaths



    def create_edge_data_for_dates(self, start_date, end_date, columns_required=None, filter_dict = None, agg_period = "week"):
        if agg_period not in ["week","month"]:
            raise ValueError("period should be either week or month")

        analysis_filepaths = self.construct_filepaths_from_dates(start_date, end_date)
        results = []
        for i in trange(len(analysis_filepaths)):
            # Read file, subset columns and filter geographical level of origin
            dat = bu.read_file(analysis_filepaths[i], columns_required, filter_dict, convert_to_date=True)
            # Construct (origin, destination, travels) information with additional data (date)
            dat = bu.get_travels_df(dat,additional_cols_req=["date_range_start"],unroll=True)
            results.append(dat)
        results = pd.concat(results)
        return bu.agg_over_time(results, period=agg_period)

    def create_node_df_from_edge_df(self, edge_df):
        return pd.DataFrame({"census_block_group": bu.get_required_fips(edge_df)})

    def create_network_with_additional_data(self, node_df, edge_df, lat_long=False, census_columns= None):
        """
        Function to add on lat-long information and census information to nodedf
        """
        required_fips = list(set(node_df["census_block_group"]))
        if lat_long:
            node_df = node_df.merge(bu.get_lat_long_data(required_fips, self.__CENSUS_PATH), 
                                    on="census_block_group", 
                                    how="left", copy=False)
        if (census_columns is not None) and len(census_columns) > 0:
            # TODO: Add validation of census columns - restrict to values in the data
            census_df, census_mapping_table = bu.get_census_data(required_fips=required_fips,
                                                required_cols=census_columns,
                                                metadata_path=self.__CENSUS_PATH)
            node_df = node_df.merge(census_df,on="census_block_group", 
                                    how="left", copy=False)
        else:
            census_mapping_table = None
            
        return {
            "node_df" : node_df,
            "edge_df" : edge_df,
            "census_mapping_table" : census_mapping_table
        }


    def get_comparison_df(self, edge_df, filter_dict, agg_period = "week", year_delta=-1, week_delta=0):
        """
        Function to create a comparison edge data frame for the given edge dataframe.
        By default, it takes same week for the previous year 
        """
        new_start_dates = edge_df["date_start"].apply(bu.comparative_date_from_date, year_delta=year_delta, week_delta=week_delta)
        new_end_dates = edge_df["date_end"].apply(bu.comparative_date_from_date, year_delta=year_delta, week_delta=week_delta)
        comparison_start_date = min(min(new_start_dates), min(new_end_dates))
        comparison_end_date = max(max(new_start_dates), max(new_end_dates))

        comparison_df = self.create_edge_data_for_dates(comparison_start_date, comparison_end_date, filter_dict = filter_dict, agg_period =agg_period)
        return comparison_df
        


    def merge_analysis_and_comparison(self, df_analysis, df_comparison, merge_key=None, 
                          merge_how="left", drop_na = False):
        fixed_cols = ["origin","destination"]
        # print(merge_key)
        if merge_key is None:
            merge_cols = fixed_cols
        else:
            if type(merge_key) == type(str()):
                merge_key = [merge_key]
            #print(merge_key)
            merge_key.extend(fixed_cols)
            merge_cols = list(set(merge_key))
        print(f'Merging on columns : {merge_cols}')
        df_combined = df_analysis.merge(df_comparison,how = merge_how,
                                    on= merge_cols, 
                                    suffixes=["_analysis","_comparison"])
        if drop_na:
            df_combined = df_combined.dropna()
        return df_combined


    def data_prep_for_Network(self, node_df, edge_df, split_edge_on=None):
    
        #node_df_cp = node_df.copy()
        #edge_df_cp = edge_df.copy()
        assert node_df.shape[0] == len(set(node_df["census_block_group"]))
        
        id2node_mapping = dict(zip(np.arange(node_df.shape[0]), node_df["census_block_group"]))
        node2id_mapping = dict(zip(node_df["census_block_group"], np.arange(node_df.shape[0])))
        
        node_df_cp = pd.DataFrame({"census_block_group" : node_df["census_block_group"].map(node2id_mapping)})
            
        edge_df_cp = pd.DataFrame({"origin" :edge_df["origin"].map(node2id_mapping),
                                "destination" : edge_df["destination"].map(node2id_mapping)})
        
        node_df_cols = list(node_df.columns)
        node_df_cols.remove("census_block_group")
        print(f"Node df columns : {node_df_cols}")
        
        edge_df_cols = list(edge_df.columns)
        edge_df_cols.remove("origin")
        edge_df_cols.remove("destination")
        print(f"Edge df columns : {edge_df_cols}")
        
        node_df_cp = pd.concat([node_df_cp, node_df[node_df_cols]], axis=1)
        edge_df_cp = pd.concat([edge_df_cp, edge_df[edge_df_cols]], axis=1)
        
        if split_edge_on is not None:
            edge_df_cp = [ edf for indx, edf in list(edge_df_cp.groupby(split_edge_on))]
        else:
            edge_df_cp = [edge_df_cp]
        
        return { "node_df" : node_df_cp,
                "edge_df" : edge_df_cp,
                "node_mapping" : id2node_mapping
            }

    def generate_networkX(self, node_df, edge_df, keep_edgeless_nodes=False):
        networks = []
        if type(edge_df) is pd.core.frame.DataFrame:
            edge_df = [edge_df]
        
        if type(edge_df) is list:
            for df in edge_df:
                net = nx.from_pandas_edgelist(df, source="origin", 
                                            target="destination", edge_attr=True, 
                                            create_using=nx.DiGraph())
                if keep_edgeless_nodes:
                    nd_df = node_df
                else:
                    nd_df = node_df[node_df["census_block_group"].isin(list(net.nodes()))]
                nd_df = node_df.set_index("census_block_group")
                nx.set_node_attributes(net, nd_df.to_dict("index"))
                networks.append(net)
        return networks
        