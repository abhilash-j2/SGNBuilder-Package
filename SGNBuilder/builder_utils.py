#builder_utils.py


from datetime import datetime, date, timedelta
import os
import sys
import pandas as pd
from pathlib import Path
import datetime
import calendar
import json
from collections import defaultdict
from tqdm import tqdm
# import pkgutil
from SGNBuilder import constants


def preprocess_FIPS(x):
    str_x = str(x).zfill(12)
    return str_x
    

def get_state_code(FIPS12digit):
    return preprocess_FIPS(FIPS12digit)[:2]

def get_county_code(FIPS12digit):
    return preprocess_FIPS(FIPS12digit)[2:5]

def get_tract_code(FIPS12digit):
    return preprocess_FIPS(FIPS12digit)[5:11]


def subset_for_state(dat, states_req, return_mask = True, fips_str_colname = "origin_census_block_group"):
    """
    Subsets the data set for the given set of state ids in FIPS code
    
    Parameters
    ----------
    dat : Pandas dataframe
    
    states_req : str or list of str containing state ids (2 digit codes)
    
    return_mask : boolean , indicating whether to return the masking vector of bools or the subsetted data
    
    fips_str_colname : str - the name of the columns containing the 12 digit string representation of FIPS code
    
    """
    # Validation checks
    if type(states_req) == type(list()):
        states_req = list(map(lambda s: pad_left(s,2), states_req))
    elif type(states_req) == type(str()):
        states_req = pad_left(states_req, 2)
        assert len(states_req) == 2, "States to filter should be either a 2 digit string or a list of ids"
        states_req = [states_req]
    else:
        raise ValueError("States to filter should be either a 2 digit string or a list of ids")
    
    # Obtaining the subseting condition as boolean
    mask = dat[fips_str_colname].apply(lambda x: get_state_code(x) in states_req)
    if return_mask:
        return mask
    else:
        return dat.loc[mask,:]

    
def subset_for_state_county(dat, state_counties_req, return_mask = True, \
                            fips_str_colname = "origin_census_block_group"):
    """
    Subsets the data set for the given set of (state id, county id) in FIPS code
    
    Parameters
    ----------
    dat : pandas.core.frame.DataFrame
    
    state_counties_req :  list,  
        list of 2-dim tuples containing (state id, county id) expected digits in id :(2, 3)
    
    return_mask : boolean, default = True 
        indicating whether to return the masking vector of bools or the subsetted data
    
    fips_str_colname : str, default = "origin_census_block_group"
        the name of the columns containing the 12 digit string representation of FIPS code
    
    """
    # Validation checks
    if type(state_counties_req) == type(list()):
        content_types = set(map(type, state_counties_req))
        type_check = (len(content_types) == 1) and \
                        ((type(tuple()) in content_types) or \
                         (type(list()) in content_types))
        content_len = set(map(len, state_counties_req))
        len_check = (2 in content_len) and len(content_len) == 1
        
        if type_check and len_check:
            state_counties_req = list(map(lambda sc: (pad_left(sc[0],2), pad_left(sc[1],3)), state_counties_req))
        else:
            raise ValueError("Filter should be a list of tuples with the format [(state_id, county_id), ..]")
        mask = dat[fips_str_colname].apply(lambda x: (get_state_code(x), get_county_code(x)) in state_counties_req)
    else:
        raise ValueError("Filter should be a list of tuples with the format [(state_id, county_id), ..]")
        
    # Obtaining the subseting condition as boolean
    if return_mask:
        return mask
    else:
        return dat.loc[mask,:]

def pad_left(item, width, pad_char="0"):
    """
    Apply padding on the left of the given item until the width is achieved
    
    Parameters
    ----------
    item : str
        input to be padded
    width : int
        width of the resultant string to pad till
    pad_char : str, default="0"
        char to use as padding
    
    """
    return str(item).rjust(width,pad_char)

def cross_join(df1, df2):
    """
    Performs a cross join on the given dataframes
    """
    df1["_key"] = 1
    df2["_key"] = 1
    return pd.merge(df1, df2,on="_key").drop(columns = "_key")

def rep_join(df1, row_df):
    """
    Performs a cross join on the given dataframes
    """
    df = df1.copy()
    for col in row_df.columns:
        df.loc[:,col] = row_df[col].item()
    return df



def parse_as_date(date_str, format_str = "%Y-%m-%d"):
    return pd.to_datetime(date_str,format= format_str)

def start_of_week(date_obj):
    date = min(date_obj)
    start_of_week = date - timedelta(days=date.dayofweek)
    return start_of_week

def end_of_week(date_obj):
    date = max(date_obj)
    end_of_week = date - timedelta(days=date.dayofweek) + timedelta(days=6) 
    return end_of_week

def start_of_month(date_obj):
    date = min(date_obj)
    first_day = date.replace(day = 1)
    return first_day

def end_of_month(date_obj):
    date = max(date_obj)
    last_day = date.replace(day = calendar.monthrange(date.year, date.month)[1])
    return last_day

def comparative_date_from_weekno(year, weekno, dayno=1):
    return date.fromisocalendar(year, weekno, dayno)

def comparative_date_from_date(dateobj, year_delta=0, week_delta=0):
    year_no, week_no, day_no = dateobj.isocalendar()
    # print( year_no, week_no, day_no)
    new_date  = date.fromisocalendar(year_no + year_delta, week_no + week_delta, day_no)
    return new_date

def get_comparative_date_ranges(min_date, max_date, year_delta=-1, week_delta=0):
    new_min_date = comparative_date_from_date(min_date, year_delta, week_delta)
    new_max_date = comparative_date_from_date(max_date, year_delta, week_delta)
    return pd.date_range(new_min_date, new_max_date, freq='d')


def apply_cutoffs(df, cutoffs_dict):
    """
    Takes cutoffs in a dict and subsets the given dataframe df
    to contain rows that have values greater than the cutoff
    
    cutoffs_dict = {
        'column_name': min_value,
        ...
    }
    """
    df_new = df.copy()
    if type(cutoffs_dict) != type(dict()):
        raise ValueError()
    
    for key, value in cutoffs_dict.items():
        df_new = df_new[df_new[key] >= value]
    
    return df_new


def get_percentage(nr_vector, dr_vector, epsilon=1e-4):
    percent = 100*(nr_vector - dr_vector+ epsilon)/ (dr_vector + epsilon)
    return percent

def get_ratio(nr_vector, dr_vector, epsilon=1e-4):
    ratio = (nr_vector + epsilon)/ (dr_vector + epsilon)
    return ratio



### Data processing functions

   

def create_edge_triplet(origin, dest_json):
    """
    This function works on one row of the dataset. 
    Taking in the origin FIPS code and the destination FIPS json string
    """
    
    outputList = []
    dest_json = json.loads(dest_json)
    for dest, count in dest_json.items():
        dest = preprocess_FIPS(dest)
        outputList.append((origin, dest, count))
    return outputList


def unroll_edge_info(dat, additional_cols_req, dropIndex = True):
    """
    When the given dataFrame has another dataFrame under the column 'edge_info'
    containing the ["origin","destination","travels"] information as rows,
    This function helps unroll the embedded dataframe to a long-format one with 
    additional column values repeated across their specific embedded dataframe.
    
    """
    
    
    unrolled_data = pd.concat([rep_join(row["edge_info"], 
                                                pd.DataFrame(dat.loc[idx,additional_cols_req]).T) 
                       for idx, row in dat.iterrows()])
    return unrolled_data.reset_index(drop=dropIndex)


def get_travels_df(dat, origin_col="origin_census_block_group", 
                   dest_json_col="destination_cbgs",
                   additional_cols_req = None,
                   unroll = False):
    
    edges = dat.apply(lambda x: create_edge_triplet(x[origin_col], x[dest_json_col]), axis = 1)

    if additional_cols_req is None:
        edges = [elem  for listObj in edges.to_list() for elem in listObj]
        edges = pd.DataFrame(edges, columns=["origin","destination","travels"])
        return edges
    else:
        if type(additional_cols_req) == type(str()):
            additional_cols_req = [additional_cols_req]
        elif type(additional_cols_req) == type(tuple()):
            additional_cols_req = list(additional_cols_req)
        elif type(additional_cols_req) != type(list()):
            raise ValueError("additional_cols_req must be a list of string values")
        
        subset_data = dat.loc[:,additional_cols_req]
        subset_data["edge_info"] = edges.apply(lambda data: pd.DataFrame(data, columns=["origin","destination","travels"]))
        
        if unroll:
            subset_data = unroll_edge_info(subset_data, additional_cols_req)

        return subset_data



def read_file(filepath, columns_required=None, filter_dict = None, convert_to_date = True):
    """
    Read the file from disk, subset for selected columns and selected rows of State, Counties
    
    Parameters
    ----------
    filepath : {str, Path}
        Path to the file to be read
    columns_required : list, default = None
        List of columns to be subsetted
    filter_dict : dict, default = None
        Dict containing the elements to filter by 
        e.g. =  { state: [], state_county : [(state, county), ..]}
        only one key is processed
    
    
    """
    fixed_columns = ['origin_census_block_group', 'date_range_start', 'destination_cbgs']
    
    if columns_required is None:
        columns_required = fixed_columns
    else:
        if type(columns_required) == type(str()):
            columns_required = [columns_required]
        columns_required.extend(fixed_columns)
        columns_required = list(set(columns_required))
    
    dat = pd.read_csv(filepath, usecols=columns_required,compression='gzip')
    ## Pre process FIPS to string
    dat["origin_census_block_group"] = dat["origin_census_block_group"].apply(preprocess_FIPS)
    if convert_to_date:
        dat["date_range_start"] = pd.to_datetime(dat["date_range_start"].apply(lambda x: x.split("T")[0]),format="%Y-%m-%d")
        if ("date_range_end" in columns_required):
            dat["date_range_end"] = pd.to_datetime(dat["date_range_end"].apply(lambda x: x.split("T")[0]),format="%Y-%m-%d")
    
    # filter_dict = { state: [], state_county : [(state, county), ..]}
    
    ## Create columns for state, county and tract on the fly
    ## Subset for the specified state or state county combo

    if filter_dict is None:
        return dat
    else:
        if "state" in filter_dict.keys():
            states_req = filter_dict["state"]
            mask = subset_for_state(dat, states_req)
            
        elif "state_county" in filter_dict.keys():
            state_counties_req = filter_dict["state_counties"]
            mask = subset_for_state_county(dat, state_counties_req)
        elif "state_county_tract" in filter_dict.keys():
            raise NotImplementedError()
        elif "fips" in filter_dict.keys():
            fips = filter_dict["fips"]
            if type(fips) == type(list()):
                mask = dat["origin_census_block_group"].apply(lambda x: x in fips)
            else:
                raise ValueError
        return dat.loc[mask,:]

def agg_over_time(df, agg_method="sum", date_col="date_range_start", period = "week"):
    # Columns expected = ["origin","destination", date_col, "travels"]
    if period == "day":
        print("No aggregation is performed as data is at a day level")
        df["day"] = date_col
        return df
    dat=df.copy()
    date_group_col = date_col
    # if period == "week":
    #     date_group_col = "week"
    #     dat[date_group_col] = dat[date_col].dt.isocalendar().week
    if period == "week":
        date_group_col = "year_week"
        dat[date_group_col] = list(zip(dat[date_col].dt.isocalendar().year, dat[date_col].dt.isocalendar().week))
    # elif period == "month":
    #     date_group_col = "month"
    #     dat[date_group_col] = dat[date_col].dt.month
    elif period == "month":
        date_group_col = "year_month"
        dat[date_group_col] = list(zip(dat[date_col].dt.isocalendar().year, dat[date_col].dt.month))
    
    print(dat.info())
    if period in {"week","year_week"}:
        agg_df = dat.groupby(by=["origin","destination",date_group_col])\
                .agg(travels = pd.NamedAgg("travels",agg_method),
                     date_start = pd.NamedAgg(date_col, start_of_week),
                     date_end  = pd.NamedAgg(date_col, end_of_week)).reset_index()
        # if period == "year_week":
        agg_df["week"] = agg_df["year_week"].apply(lambda x:x[1])
    elif period in {"month", "year_month"}:
        agg_df = dat.groupby(by=["origin","destination",date_group_col])\
                .agg(travels = pd.NamedAgg("travels",agg_method),
                     date_start = pd.NamedAgg(date_col, start_of_month),
                     date_end  = pd.NamedAgg(date_col, end_of_month)).reset_index()
        # if period == "year_month":
        agg_df["month"] = agg_df["year_month"].apply(lambda x:x[1])
    return agg_df



def get_required_fips(edge_df):
    fips = set()
    fips.update(edge_df["origin"])
    fips.update(edge_df["destination"])
    return list(fips)



def get_lat_long_data(required_fips, metadata_path,
                      cols_required = ["census_block_group","latitude","longitude"] 
                      ):
    
    required_cols = set(["census_block_group","latitude","longitude"])
    required_cols.update(cols_required)
    
    latlong_df = pd.read_csv(Path(metadata_path,"cbg_geographic_data.csv"), usecols=required_cols)
    # Preprocess and make fips code a 12-digit string and zeropadded
    latlong_df["census_block_group"] = latlong_df["census_block_group"].apply(preprocess_FIPS)
    latlong_df = latlong_df[latlong_df["census_block_group"].isin(required_fips)]
    latlong_df = latlong_df.reset_index(drop=True)
    print(latlong_df.info())
    # print(required_fips)
    return latlong_df

def subset_census_file(filename, required_cols, required_fips):
    df = pd.read_csv(filename, usecols=required_cols)
    df["census_block_group"] = df["census_block_group"].apply(preprocess_FIPS)
    df = df[df["census_block_group"].isin(required_fips)]
    return df

def get_census_data(required_fips, required_cols, metadata_path):
    census_field_desc = pd.read_csv(Path(metadata_path, "cbg_field_descriptions.csv"))
            
    files2cols_required = defaultdict(list)
    # Obtain files and fields within those files that match the required cols
    for col_desc in required_cols:
        field = constants.census_desc2field[col_desc]
        filename = constants.census_fields2files[field]
        files2cols_required[filename].append(field)
    
    df = pd.DataFrame({"census_block_group": list(required_fips)})
    
    # Adding primary key to each file's columns required
    for file in files2cols_required.keys():     
        files2cols_required[file].append("census_block_group")
        print("Merging file", file)
        df = df.merge(subset_census_file(filename=Path(metadata_path,"../data/",file), 
                                         required_cols=files2cols_required[file],
                                         required_fips=required_fips), 
                      copy=False, on="census_block_group")
    mapping_table = census_field_desc[census_field_desc["table_id"].isin(df.columns)]
    return df, mapping_table


def fill_comparison_na(combined_df, year_delta=-1, week_delta=0):
    df = combined_df.copy()
    df["travels_comparison"] = df["travels_comparison"].where(~df["travels_comparison"].isna(), 0)
    df["date_start_comparison"] = pd.to_datetime(df["date_start_analysis"].\
                                                 apply(comparative_date_from_date, year_delta=year_delta, week_delta=week_delta))
    df["date_end_comparison"] =   pd.to_datetime(df["date_end_analysis"].\
                                                 apply(comparative_date_from_date, year_delta=year_delta, week_delta=week_delta))
    df["year_week_comparison"] = list(zip(df["date_start_comparison"].dt.isocalendar().year, 
                                          df["date_start_comparison"].dt.isocalendar().week))
    return df
    
