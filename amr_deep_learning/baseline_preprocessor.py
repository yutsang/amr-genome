import os
import sys
import itertools
import glob
import subprocess
from multiprocessing import Pool
from tqdm import tqdm

def get_all_file_name(rpath_raw):
    # list all the files in rpath_raw
    files = os.listdir(rpath_raw)
    return files

def generate_kmer_hash(k, folder, filename, output_folder):
    prefix = filename.strip(".fna")
    filepath = folder + filename
    subprocess.run(["jellyfish count " + "-m " + k + " -s " + "1000000 " + " -t 32 "+ "-o " + output_folder \
        + prefix + ".jf " + filepath], shell=True)

def generate_kmer_hash_wrapper(args):
    return generate_kmer_hash(*args)

from multiprocessing import cpu_count

def generate_vector_single_file(args):
    file, output_folder = args
    base_name = file.rsplit('.jf_0', 1)[0]
    os.rename(file, base_name + '.jf')
    prefix = base_name.split('/')[-1]
    filepath = base_name + '.jf'
    subprocess.run(["jellyfish query " +  filepath + " < ./query.fasta > " + output_folder + prefix + ".txt "], shell=True)
    with open(output_folder + prefix + ".txt", 'r') as fd:
        lines = fd.readlines()
    lines = [' '.join(line.split())+'\n' for line in lines]
    with open(output_folder + prefix + ".txt", 'w') as fd:
        fd.writelines(lines)
    os.remove(filepath)  # This line removes the .jf file

def generate_vector(folder, output_folder):
    files = glob.glob(folder + '*.jf_0')
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(generate_vector_single_file, [(file, output_folder) for file in files]), total=len(files)))

def main():
    k = sys.argv[1]
    rpath_raw = sys.argv[2]
    file_names = get_all_file_name(rpath_raw)

    subprocess.run(["mkdir","-p","K-mers"])

    query = ""
    fd = open("./query.fasta",'w')
    for i in itertools.product('ATCG', repeat=int(k)):
        query = "".join(i)
        fd.write(f"{query}\n")  # Removed the duplicate line here
    fd.close()

    with Pool() as p:
        list(tqdm(p.imap(generate_kmer_hash_wrapper, [(k, rpath_raw, filename, "./K-mers/") for filename in file_names]), total=len(file_names)))

    generate_vector("./K-mers/", "./K-mers/")

if __name__ == '__main__':
    main()