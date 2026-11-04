"""Run with: python -m unittest test_postprocess.py"""
import unittest
from postprocess import suppress_duplicates


class TestNMS(unittest.TestCase):
    def test_duplicate_keeps_highest_score_without_mutating_input(self):
        original={'img':[{'box':[0,0,10,10],'score':.4},{'box':[0,0,10,10],'score':.9}]}
        result=suppress_duplicates(original,.5)
        self.assertEqual(len(result['img']),1)
        self.assertEqual(result['img'][0]['score'],.9)
        self.assertEqual(len(original['img']),2)
        result['img'][0]['box'][0]=100
        self.assertEqual(original['img'][1]['box'][0],0)

    def test_separate_and_exact_boundary_objects_are_kept(self):
        items=[{'box':[0,0,10,10],'score':.9},{'box':[0,0,5,10],'score':.8},
               {'box':[30,30,40,40],'score':.7}]
        self.assertEqual(len(suppress_duplicates({'img':items},.5)['img']),3)

    def test_empty_images_are_preserved_and_invalid_boxes_rejected(self):
        result=suppress_duplicates({'empty':[],'bad':[{'box':[1,1,0,0],'score':.8}]})
        self.assertEqual(result,{'empty':[],'bad':[]})
        with self.assertRaises(ValueError):suppress_duplicates({},1.1)


if __name__=='__main__':unittest.main()
