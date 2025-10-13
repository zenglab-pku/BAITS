import anndata


def _save_data(adata, attr, key, data):
    obj = getattr(adata, attr)
    obj[key] = data





