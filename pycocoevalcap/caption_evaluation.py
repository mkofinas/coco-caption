from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import zip  # pylint: disable=redefined-builtin
from builtins import dict  # pylint: disable=redefined-builtin
from builtins import object  # pylint: disable=redefined-builtin

from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from cider.cider import Cider


class CaptionEvaluation(object):
    def __init__(self, image_ids, gts, res):
        self.evalImgs = []
        self.evaluation_scores = {}
        self.imgToEval = dict()
        self.params = {'image_id': image_ids}
        self.gts = gts
        self.res = res

    def evaluate(self):
        imgIds = self.params['image_id']
        gts = self.gts
        res = self.res

        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        print(gts, res)

        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]

        for scorer, method in scorers:
            print('computing {0} score...'.format(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self._setEval(sc, m)
                    self._setImgToEvalImgs(scs, imgIds, m)
                    print("{0}: {1:.3f}".format(m, sc))
            else:
                self._setEval(score, method)
                self._setImgToEvalImgs(scores, imgIds, method)
                print("{0}: {1:.3f}".format(method, score))
        self._setEvalImgs()

    def _setEval(self, score, method):
        self.evaluation_scores[method] = score

    def _setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = dict([("image_id", imgId)])
            self.imgToEval[imgId][method] = score

    def _setEvalImgs(self):
        self.evalImgs = list(self.imgToEval.values())


def calculate_metrics(image_ids, dataset_gts, dataset_res):
    imgToAnnsGTS = {ann['image_id']: [] for ann in dataset_gts['annotations']}
    for ann in dataset_gts['annotations']:
        imgToAnnsGTS[ann['image_id']].append(ann)

    imgToAnnsRES = {ann['image_id']: [] for ann in dataset_res['annotations']}
    for ann in dataset_res['annotations']:
        imgToAnnsRES[ann['image_id']].append(ann)

    gts = {}
    res = {}
    for img_id in image_ids:
        gts[img_id] = imgToAnnsGTS[img_id]
        res[img_id] = imgToAnnsRES[img_id]

    evalObj = CaptionEvaluation(image_ids, gts, res)
    evalObj.evaluate()
    return evalObj.evaluation_scores


if __name__ == '__main__':
    image_ids = [0, 1]
    dataset_gts = {
        'annotations': [
            {u'image_id': 0, u'caption': u'the man is playing a guitar'},
            {u'image_id': 0, u'caption': u'a man is playing a guitar'},
            {u'image_id': 1, u'caption': u'a woman is slicing cucumbers'},
            {u'image_id': 1, u'caption': u'the woman is slicing cucumbers'},
            {u'image_id': 1, u'caption': u'a woman is cutting cucumbers'}]
        }
    dataset_res = {
        'annotations': [
            {u'image_id': 0, u'caption': u'man is playing guitar'},
            {u'image_id': 1, u'caption': u'a woman is cutting vegetables'}]
        }
    print(calculate_metrics(image_ids, dataset_gts, dataset_res))
    from IPython import embed
    embed()
