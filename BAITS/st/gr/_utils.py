import anndata


def _save_data(adata, attr, key, data):
    """
    Save data into a specified attribute of an AnnData object.

    This is a lightweight helper function used to store results
    into an attribute container of an AnnData object, such as
    ``adata.obs``, ``adata.var``, ``adata.obsm``, or ``adata.uns``.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    attr : str
        Attribute name of the AnnData object where the data will be stored.
        For example: ``"obs"``, ``"var"``, ``"obsm"``, or ``"uns"``.
    key : str
        Key under which the data will be stored in the selected attribute.
    data : Any
        Data object to store. The type depends on the attribute target
        (e.g., pandas Series/DataFrame, numpy array, or dict).

    Returns
    -------
    None
        The function modifies the AnnData object in place.

    Notes
    -----
    This function directly updates the specified AnnData attribute
    by assigning ``adata.<attr>[key] = data``.
    """
    obj = getattr(adata, attr)
    obj[key] = data


