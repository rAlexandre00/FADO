from abc import abstractmethod


class Downloader(object):

    @abstractmethod
    def download(self):
        pass
