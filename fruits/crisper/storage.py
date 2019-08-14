import os
import time

import mandalka

from . import RESULTS_PATH, CACHE_PATH, TRASH_PATH, LOGGER, safe_path_join

class LockedError(Exception):
    pass

class MandalkaStorage():

    def __init__(self, *args, **kwargs):
        while True:
            if self.mandalka_exists():
                self.mandalka_load(*args, **kwargs)
                mandalka.del_arguments(self)
                return
            else:
                try:
                    self.mandalka_lock()
                    if self.mandalka_exists():
                        LOGGER.warning("don't worry...")
                        self.mandalka_unlock()
                        continue
                    failed = True
                    try:
                        assert not self.mandalka_exists()
                        self.log("Building: {}...".format(self))
                        self.mandalka_build(*args, **kwargs)
                        self.mandalka_save_cache()
                        self.mandalka_clean_after_build()
                        self.log("Ready: {}".format(self))
                        failed = False
                    finally:
                        if failed is True:
                            self.mandalka_unlock()
                            raise
                        else:
                            self.mandalka_save_and_unlock()
                except LockedError:
                    LOGGER.warning("another process evaluating node {}, waiting 10 seconds...".format(self))
                    time.sleep(10)

    @mandalka.lazy
    def mandalka_results_path(self):
        return safe_path_join(mandalka.unique_id(self), dirname=RESULTS_PATH)

    @mandalka.lazy
    def mandalka_cache_path(self):
        return safe_path_join(mandalka.unique_id(self) + ".tmp", dirname=CACHE_PATH)

    @mandalka.lazy
    def mandalka_trash_path(self):
        return safe_path_join(mandalka.unique_id(self), dirname=TRASH_PATH)

    def mandalka_lock(self):
        try:
            os.makedirs(self.mandalka_cache_path())
        except FileExistsError:
            raise LockedError()

    def mandalka_unlock(self):
        os.rmdir(self.mandalka_cache_path())

    @mandalka.lazy
    def mandalka_locked(self):
        return os.path.exists(self.mandalka_cache_path())

    @mandalka.lazy
    def mandalka_exists(self):
        return os.path.exists(self.mandalka_results_path())

    @mandalka.lazy
    def mandalka_remove(self):
        os.rename(
            self.mandalka_results_path(),
            self.mandalka_trash_path())

    def mandalka_save_and_unlock(self):
        os.rename(
            self.mandalka_cache_path(),
            self.mandalka_results_path())

    def mandalka_build(self, *args, **kwargs):
        raise NotImplementedError()

    def mandalka_save_cache(self):
        raise NotImplementedError()

    def mandalka_clean_after_build(self):
        raise NotImplementedError()

    def mandalka_load(self, *args, **kwargs):
        raise NotImplementedError()

    def log(self, msg):
        LOGGER.info(msg)
