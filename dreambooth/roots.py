import importlib.metadata

def print_package_roots(pkg='transformer'):
    for dist in importlib.metadata.distributions():
        if pkg in dist.metadata['Name']:
            print(f"{dist.metadata['Name']}: {dist.locate_file('')}")

print_package_roots()