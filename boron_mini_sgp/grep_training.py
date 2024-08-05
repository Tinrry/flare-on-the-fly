def write_lattice(read_file, save_file):
    num_atoms = int(read_file.readline().strip())
    save_file.write(str(num_atoms) + '\n')
    for _ in range(num_atoms+1):
        line = read_file.readline()
        save_file.write(line)       # already has '\n'
    return num_atoms


num_atoms = 0
num_lattice = 10
with open('./training.xyz', 'w') as save_file:
    with open('./training_set.xyz', 'r') as read_file:
        for _ in range(num_lattice):
            num_atoms = write_lattice(read_file, save_file)
            print(f'added {num_atoms} atoms')
