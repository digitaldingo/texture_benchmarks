import os
import shutil
import tempfile
import logging

from .file_util import checksum, download, archive_extract, is_archive, touch


logger = logging.getLogger(__name__)

_CHECKPOINT_FILE = '__download_complete'


class BaseDataset:
    _data_root = os.environ['TEXTURE_DATA_ROOT']

    def _download_data(self, urls, target_dir, sha1s=None):
        ''' Dowload dataset files given by the urls. Archive files are
        extracted automatically. If sha1s is specified, the downloaded files
        are checked for correctness.'''
        target_dir = os.path.abspath(target_dir)
        checkpoint = os.path.join(target_dir, _CHECKPOINT_FILE)
        if os.path.exists(checkpoint):
            # Dataset is already on disk.
            return
        if os.path.exists(target_dir):
            logger.info('Incomplete dataset %s exists - restarting download.'
                        % target_dir)
            shutil.rmtree(target_dir)
            os.mkdir(target_dir)
        else:
            os.makedirs(target_dir)
        for i, url in enumerate(urls):
            logger.info('Downloading %s' % url)
            filepath = download(url, target_dir)
            if sha1s is not None:
                if sha1s[i] != checksum(filepath):
                    raise RuntimeError('SHA-1 checksum mismatch for %s.'
                                       % url)
            if is_archive(filepath):
                logger.info('Extracting %s' % filepath)
                archive_extract(filepath, target_dir)
                os.remove(filepath)
        touch(checkpoint)

    def _url_checksums(self, urls):
        ''' Utility function for generating SHA-1 checksums for a dataset.'''
        temp_dir = tempfile.mkdtemp()
        checksums = []
        for url in urls:
            logger.info('Generating checksum for %s' % url)
            filepath = download(url, temp_dir)
            checksums.append(checksum(filepath))
            os.remove(filepath)
        return checksums
