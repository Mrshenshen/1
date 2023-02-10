# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def calculate_similarity(image_embedding, text_embedding):
    image_embedding = image_embedding.view(image_embedding.size(0), -1)
    text_embedding = text_embedding.view(text_embedding.size(0), -1)
    image_embedding_norm = image_embedding / (image_embedding.norm(dim=1, keepdim=True) + 1e-8)
    text_embedding_norm = text_embedding / (text_embedding.norm(dim=1, keepdim=True) + 1e-8)

    similarity = torch.mm(image_embedding_norm, text_embedding_norm.t())

    similarity_match = torch.sum(image_embedding_norm * text_embedding_norm, dim=1)

    return similarity, similarity_match


def calculate_margin_cr(similarity_match_cr, similarity_match, auto_margin_flag, margin):
    if auto_margin_flag:
        lambda_cr = abs(similarity_match_cr.detach()) / abs(similarity_match.detach())
        ones = torch.ones_like(lambda_cr)
        data = torch.ge(ones, lambda_cr).float()
        data_2 = torch.ge(lambda_cr, ones).float()
        lambda_cr = data * lambda_cr + data_2

        lambda_cr = lambda_cr.detach().cpu().numpy()
        margin_cr = ((lambda_cr + 1) * margin) / 2.0
    else:
        margin_cr = margin / 2.0

    return margin_cr


class CRLoss(nn.Module):

    def __init__(self, opt):
        super(CRLoss, self).__init__()

        self.device = opt.device
        self.margin = np.array([opt.margin]).repeat(opt.batch_size)
        self.double_margin = np.array([opt.margin]).repeat(opt.batch_size*2*opt.part)
        self.beta = opt.cr_beta
        # self.margin_local = np.array([opt.margin]).repeat(opt.batch_size*opt.part)

    def semi_hard_negative(self, loss, margin):
        negative_index = np.where(np.logical_and(loss < margin, loss > 0))[0]
        return np.random.choice(negative_index) if len(negative_index) > 0 else None

    def get_triplets(self, similarity, labels, auto_margin_flag, margin):

        similarity = similarity.cpu().data.numpy()

        labels = labels.cpu().data.numpy()
        triplets = []

        for idx, label in enumerate(labels):  # same class calculate together
            if margin[idx] >= 0.16 or auto_margin_flag is False:
                negative = np.where(labels != label)[0]

                ap_sim = similarity[idx, idx]

                loss = similarity[idx, negative] - ap_sim + margin[idx]

                negetive_index = self.semi_hard_negative(loss, margin[idx])

                if negetive_index is not None:
                    triplets.append([idx, idx, negative[negetive_index]])

        if len(triplets) == 0:
            triplets.append([idx, idx, negative[0]])

        triplets = torch.LongTensor(np.array(triplets))

        return_margin = torch.FloatTensor(np.array(margin[triplets[:, 0]])).to(self.device)

        return triplets, return_margin

    def calculate_loss(self, similarity, label, auto_margin_flag, margin):

        image_triplets, img_margin = self.get_triplets(similarity, label, auto_margin_flag, margin)
        text_triplets, txt_margin = self.get_triplets(similarity.t(), label, auto_margin_flag, margin)

        image_anchor_loss = F.relu(img_margin
                                   - similarity[image_triplets[:, 0], image_triplets[:, 1]]
                                   + similarity[image_triplets[:, 0], image_triplets[:, 2]])

        similarity = similarity.t()
        text_anchor_loss = F.relu(txt_margin
                                  - similarity[text_triplets[:, 0], text_triplets[:, 1]]
                                  + similarity[text_triplets[:, 0], text_triplets[:, 2]])

        loss = torch.sum(image_anchor_loss) + torch.sum(text_anchor_loss)

        return loss

    def forward(self, img, txt, txt_cr, labels, auto_margin_flag,local=False):
        # if local:
        #     similarity, similarity_match = calculate_similarity(img, txt)
        #     similarity_cr, similarity_cr_match = calculate_similarity(img, txt_cr)
        #     margin_cr = calculate_margin_cr(similarity_cr_match, similarity_match, auto_margin_flag, self.double_margin)
        #
        #     cr_loss = self.calculate_loss(similarity, labels, auto_margin_flag, self.double_margin) \
        #               + self.beta * self.calculate_loss(similarity_cr, labels, auto_margin_flag, margin_cr)
        # else:
        similarity, similarity_match = calculate_similarity(img, txt)
        similarity_cr, similarity_cr_match = calculate_similarity(img, txt_cr)
        margin_cr = calculate_margin_cr(similarity_cr_match, similarity_match, auto_margin_flag, self.margin)

        cr_loss = self.calculate_loss(similarity, labels, auto_margin_flag, self.margin) \
                  + self.beta * self.calculate_loss(similarity_cr, labels, auto_margin_flag, margin_cr)

        return cr_loss

def cosine_sim(im, s):
    """Cosine similarity between all the two pairs
    """
    return im.mm(s.t())


def l1_sim(im, s):
    """l1 similarity between two pairs
    """
    scro = torch.cdist(im, s, p=1)
    return scro


def l2_sim(im, s):
    """L2 similarity between two pairs
    """
    scro = torch.cdist(im, s, p=2)
    return scro


def msd_loss(im, s):
    """MSD similarity between two pairs
    """
    scro = torch.cdist(im, s, p=2)
    return scro.pow(2)

class IntraLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, measure=False, max_violation=False, up=0.45, down=0.1, lamb=1.0):
        super(IntraLoss, self).__init__()
        self.device = opt.device
        self.margin = margin
        self.measure = measure
        if measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'msd':
            self.sim = msd_loss
        elif self.measure == 'l1':
            self.sim = l1_sim
        elif self.measure == 'l2':
            self.sim = l2_sim
        self.max_violation = max_violation
        self.up = up
        self.down = down
        self.lamb = lamb

    def forward(self, img_emb, text_emb):
        # compute image-sentence score matrix
        mx, mx1 = calculate_similarity(img_emb, text_emb)
        scores = self.sim(mx, mx)
        #print('score1', scores)
        #print('norm:', torch.nn.functional.normalize(scores))

        if self.measure == 'cosine':
            diagonal = scores.to(self.device).diag()
            print('d1', diagonal)
            scores = scores.to(self.device)
            print('score1', scores)
            eye = torch.eye(scores.size(0)).float().to(self.device)
            scores_non_self = scores - eye
            print('score2', scores_non_self)
            # scores_non_self.gt_(self.up).lt_(1 - self.down)
            scores_non_self = scores_non_self * (
                scores_non_self.gt(self.up).float())
            scores_non_self = scores_non_self * (
                scores_non_self.lt(1 - self.down).float())
            scores_norm = scores_non_self.sum() / scores.size(0)
            #print('size:', scores.size(0))
            #scores_norm = scores_non_self.sum()

        elif self.measure == 'msd' or self.measure == 'l1' or self.measure == 'l2':
            scores_non_self = torch.nn.functional.normalize(scores).to(self.device)
            #scores_non_self = (scores).to(self.device)
            #print('score1', scores_non_self)
        
            idx_up = round(self.up * scores.size(0))
            idx_down = round(self.down * scores.size(0))
            _, s_index = scores_non_self.sort()
            s_mean = scores_non_self.mean()

            s_up = scores_non_self[0, s_index[0, idx_up]]
            s_down = scores_non_self[0, s_index[0, idx_down]]
            #print('score', scores_non_self)

            scores_non_self = scores_non_self * (
                scores_non_self.gt(s_down).float())
            scores_non_self = scores_non_self * (
                scores_non_self.lt(s_up).float())
            scores_norm = scores_non_self.sum() / scores.size(0)
            #print('size:', scores_norm)

        return self.lamb * scores_norm