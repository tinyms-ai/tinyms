from ..hub.utils.download import url_exist
import hashlib
import urllib
from urllib.request import urlretrieve, HTTPError, URLError
import pathlib
import tqdm


def sha256sum(file_name):
    fp = open(file_name, 'rb')
    content = fp.read()
    fp.close()
    m = hashlib.sha256()
    m.update(content)
    sha256 = m.hexdigest()
    return sha256


def is_directory_empty(path):
    path = pathlib.Path(path)
    if not path.exists():
        return True
    if any(path.iterdir()):
        return False
    return True


def glob_files(path):
    path = pathlib.Path(path)
    all_files = [str(i.relative_to(path)) for i in path.rglob('*') if i.is_file()]
    sha256 = [sha256sum(str(path / i)) for i in all_files]
    return list(zip(all_files, sha256))


def make_filelist(path):
    path = pathlib.Path(path)
    file_sha256 = glob_files(path)
    file_sha256 = [' '.join(i) for i in file_sha256]
    with (path / 'filelist.txt').open('w') as f:
        f.write('\n'.join(file_sha256))


class RepoDownloader:
    __prefix__ = "https://kaiyuanzhixia.obs.cn-east-3.myhuaweicloud.com/"
    __filelist__ = "filelist.txt"

    def __init__(self, path, repo="lenet5/model", force_download=False, checkfiles=True):
        self.repo = repo
        self.path = pathlib.Path(path)
        if repo is not None:
            self.path = self.path / repo
        self.force_download = force_download
        self.checkfiles = checkfiles

    def append_sys_path(self):
        import sys
        sys.path.append(str(self.path))

    def download(self):
        if not self.checkfiles:
            return
        if not self.validate_repo():
            if self.repo is None:
                _info = self.path
            else:
                _info = self.repo
            raise RuntimeError(f"Invalid repo: {_info}")

        if not is_directory_empty(self.path):
            if not self.filelist.exists():
                if not self.force_download:
                    raise RuntimeError(f"repo cached is broken: {self.path}")
                else:
                    return
            else:
                if not self.force_download:
                    return

        download_file_from_url(
            self.get_file_url(self.__filelist__), save_path=self.filelist)

        self.filter_files()
        self._download()

    def validate_repo(self):
        if not self.checkfiles:
            return True
        if self.repo is None:
            return self.filelist.exists()

        return url_exist(self.get_file_url(self.__filelist__))

    @property
    def repo_url(self):
        return self.__prefix__ + f"{self.repo}"

    def get_file_url(self, filename):
        if self.repo is None:
            return None
        return self.repo_url + "/" + filename

    def get_file_save_path(self, filename):
        return self.path / filename

    def _download(self):
        file_sha256 = self.all_files_sha256
        for f, s in tqdm.tqdm(file_sha256):
            download_file_from_url(
                self.get_file_url(f), hash_sha256=s, save_path=self.get_file_save_path(f))

    @property
    def filelist(self):
        return self.path / self.__filelist__

    @property
    def all_files_sha256(self):
        all_files = []
        with self.filelist.open() as f:
            for line in f.readlines():
                file_sha256 = line.strip().split()
                assert len(file_sha256) <= 2

                if len(file_sha256) == 1:
                    file_sha256.append(None)
                all_files.append(file_sha256)

        return all_files

    def filter_files(self):
        file_sha256 = glob_files(self.path)
        file_sha256 = dict(file_sha256)
        all_files_sha256 = dict(self.all_files_sha256)
        exclude_files = []
        exclude_files.append(str(self.filelist.relative_to(self.path)))

        for f in file_sha256:
            if f not in exclude_files:
                if f not in all_files_sha256:
                    (self.path / f).unlink()
                else:
                    if file_sha256[f] != all_files_sha256[f]:
                        (self.path / f).unlink()


class RepoDownloaderWithCode:
    def __init__(self, path, repo="lenet5/model", force_download=False, checkfiles=True):
        self.repo = repo
        self.path = pathlib.Path(path)
        self.force_download = force_download
        self.checkfiles = checkfiles

        self.model_repo_downloader = RepoDownloader(
            self.path, repo=self.repo, force_download=self.force_download)
        assert self.model_repo_downloader.validate_repo(), f"Invalid repo: {repo}"

        if repo is None:
            self.code_repo_downloader = RepoDownloader(
                str(self.path) + "_code", repo=self.repo, force_download=self.force_download)
        else:
            self.code_repo_downloader = RepoDownloader(
                self.path, repo=self.repo + "_code", force_download=self.force_download)

    def download(self):
        self.model_repo_downloader.download()

        if self.code_repo_downloader.validate_repo():
            self.code_repo_downloader.download()
            self.code_repo_downloader.append_sys_path()


def download_file_from_url(url, hash_sha256=None, save_path='.'):
    def reporthook(a, b, c):
        percent = a * b * 100.0 / c
        percent = 100 if percent > 100 else percent
        if c > 0:
            print("\rDownloading...%5.1f%%" % percent, end="")

    save_path = pathlib.Path(save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    if not save_path.exists():
        if url is None:
            raise RuntimeError("A valid repo is not given.")
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urlretrieve(url, str(save_path), reporthook=reporthook)
        except HTTPError as e:
            raise Exception(e.code, e.msg, url)
        except URLError as e:
            raise Exception(e.errno, e.reason, url)

        # Check file integrity
        if hash_sha256:
            result = sha256sum(save_path)
            result = result == hash_sha256
            if not result:
                raise Exception('INTEGRITY ERROR: File: {} is not integral'.format(save_path))


if __name__ == '__main__':
    a = RepoDownloaderWithCode('cache_dir', force_download=True)
    a.download()
