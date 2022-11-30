import torch.nn as nn


class TripletLoss_margins(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self):
        super(TripletLoss_margins, self).__init__()
        self.margins = [.1, .085, .07, .04]
        
        
    def forward(self, anchors, poss, negs, cs):
        total_loss = []
        Loss = nn.TripletMarginLoss(margin=self.margins[cs[0]], p=2)
        for i in range(len(anchors)):
            Loss.margin = self.margins[cs[i]]
            anchor = anchors[i].view(1, -1)
            pos = poss[i].view(1, -1)
            neg = negs[i].view(1, -1)
            loss = Loss(anchor, pos, neg)
            total_loss.append(loss)
            
        return sum(total_loss)/len(total_loss) 

    
if __name__ == '__main__':
    pass
        
