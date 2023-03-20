__version__ = 'v0.0.8rc1'

if __name__ == '__main__':
    print(__version__)
    import os, sys
    _, cmd, *args = sys.argv
    print(cmd, args)
    if cmd == 'tag':
        os.system(f'git tag {__version__}')
        os.system(f'git push -u origin {__version__}')
