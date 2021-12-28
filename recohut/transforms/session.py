# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/transforms/session.ipynb (unless otherwise specified).

__all__ = ['construct_session_sequences']

# Cell
import numpy as np

# Cell
def construct_session_sequences(df, sessionID, itemID):
    """
    Given a dataset in pandas df format, construct a list of lists where each sublist
    represents the interactions relevant to a specific session, for each sessionID.
    These sublists are composed of a series of itemIDs (str) and are the core training
    data used in the Word2Vec algorithm.
    This is performed by first grouping over the SessionID column, then casting to list
    each group's series of values in the ItemID column.
    INPUTS
    ------------
    df:                 pandas dataframe
    sessionID: str      column name in the df that represents invididual sessions
    itemID: str         column name in the df that represents the items within a session
    """
    grp_by_session = df.groupby([sessionID])

    session_sequences = []
    for name, group in grp_by_session:
        session_sequences.append(list(group[itemID].values))

    return session_sequences