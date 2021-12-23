# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/datasets/diginetica.ipynb (unless otherwise specified).

__all__ = ['DigineticaDataset']

# Cell
class DigineticaDataset(SessionDatasetv2):
    url = 'https://github.com/RecoHut-Datasets/diginetica/raw/main/train-item-views.csv'

    def __init__(self, root, column_names={'SESSION_ID':'sessionId',
                                        'ITEM_ID': 'itemId',
                                        'TIMEFRAME': 'timeframe',
                                        'EVENT_DATE': 'eventdate'}):
        super().__init__(root, column_names)

    @property
    def raw_file_names(self) -> str:
        return 'train-item-views.csv'

    def download(self):
        path = download_url(self.url, self.raw_dir)