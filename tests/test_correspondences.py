import unittest
import numpy as np
import kapture
import path_to_kapture_localization  # noqa: F401, to set path
from kapture_localization.localization.correspondences import get_correspondences
from kapture_localization.localization.DuplicateCorrespondencesStrategy import DuplicateCorrespondencesStrategy
from kapture_localization.localization.RerankCorrespondencesStrategy import RerankCorrespondencesStrategy
from kapture.io.tar import TarCollection
from unittest import mock

class TestCorrespondences(unittest.TestCase):
    def test_same_3d_multiple_2d(self):
        # minimal kapture with one map image and one query image
        k_data = kapture.Kapture(
            matches={'kp': kapture.Matches()},
            points3d=kapture.Points3d(np.array([[0.0, 0.0, 0.0]]))
        )
        k_data.matches['kp'].add('query', 'map')

        # mapping of map keypoints to 3D point (both keypoints refer to same 3D id)
        point_id_from_obs = {('map', 0): 0, ('map', 1): 0}

        kpts_query = np.zeros((3, 2))

        fake_matches = np.array([[0, 0, 1.0], [1, 1, 1.0]], dtype=float)

        with mock.patch('kapture_localization.localization.correspondences.get_matches_fullpath',
                        return_value='dummy'), \
             mock.patch('kapture_localization.localization.correspondences.image_matches_from_file',
                        return_value=fake_matches):
            _, _, _, stats = get_correspondences(
                k_data, 'kp', '', TarCollection(), 'query', ['map'],
                point_id_from_obs, kpts_query, None,
                DuplicateCorrespondencesStrategy.keep,
                RerankCorrespondencesStrategy.none)

        self.assertEqual(stats['same_3d_multiple_2d_count'], 1)
        self.assertEqual(stats['same_3d_multiple_2d_max'], 2)

if __name__ == '__main__':
    unittest.main()
