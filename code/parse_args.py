import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='VampPrior-Prototype')

    parser.add_argument('--seed', 
                       default=1000, 
                       type=int)
    
    parser.add_argument('--K', 
                       default=50, 
                       type=int)
    
    parser.add_argument('--model_type', 
                       default='vampprior', 
                       help='vampprior or standard',
                       type=str)
    
    parser.add_argument('--dataset', 
                       default='mnist', 
                       help='mnist or cifar10',
                       type=str)
    
    parser.add_argument('--rm_labels', 
                       default=0, 
                       type=int)

    args = parser.parse_args()
    return args