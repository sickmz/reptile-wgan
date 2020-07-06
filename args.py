import argparse

def get_args():

    # General params
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')
    parser._optionals.title = "option"

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default=16)', metavar='')
    parser.add_argument('--dataset_path', type=str, help='dataset path', default='/data/', metavar='')
    parser.add_argument('--nz', type=int, default=100, help='number of dimensions for input noise', metavar='')
    parser.add_argument('--epochs', type=int, default=100000, help='number of training epochs', metavar='')    
    parser.add_argument('--cuda', action='store_false', help='enable cuda')
    parser.add_argument('--save_every', type=int, default=5000, help='after how many epochs save the model', metavar='')
    parser.add_argument('--save_dir', type=str, default='./models/', help='path to save the trained models', metavar='')
    parser.add_argument('--samples_dir', type=str, default='./samples/', help='path to save the dataset samples', metavar='') 
    parser.add_argument('--output', type=str, default='./output/', help='path to save the images generated in the test phase', metavar='') 

    # Training params
    parser.add_argument('--classes', default=5, type=int, help='classes in base-task (n-way)', metavar='')
    parser.add_argument('--shots', default=5, type=int, help='shots per class (k-shot)', metavar='')
    parser.add_argument('--iterations', default=10, type=int, help='number of base iterations', metavar='')
    parser.add_argument('--test-iterations', default=100, type=int, help='number of test iterations', metavar='')
    parser.add_argument('--meta-lr', default=0.01, type=float, help='meta learning rate', metavar='')
    parser.add_argument('--lr', default=0.0001, type=float, help='base learning rate', metavar='')  
    
    args = parser.parse_args()
    return args