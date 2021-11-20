import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers")
    parser.add_argument('intergers', metavar='N', type=int, help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, \
                        help='sum the intergers (default: find the max')

    args = parser.parse_args()
    print(args.accumulate(args.integers))
