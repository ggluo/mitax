import importlib
import os
import yaml
import numpy as np
import tempfile
import subprocess
from datetime import datetime
import jax
import jax.numpy as jnp
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import mmap
import re
from mitax import sampler

##################################################
#      Utility Functions for training loop       #
##################################################

def get_class_by_name(module, class_name):
    """
    Dynamically imports a module and retrieves a class by name.

    Args:
        module (str): The name of the module to import.
        class_name (str): The name of the class to retrieve.

    Returns:
        class: The class object.

    Raises:
        ValueError: If the class is not found in the module or if the module does not exist.
    """
    try:
        # Import the module dynamically
        if type(module) == str:
            module = importlib.import_module(module)

        # Get the class by name from the module and return it
        return getattr(module, class_name)
    
    except (ImportError, AttributeError):
        raise ValueError(f"Class {class_name} not found in module {module} or module {module} does not exist.")


def load_config(path):
    """
    Load configuration defined with a YAML file.

    Args:
        path (str): The path to the YAML file.

    Returns:
        dict: The loaded configuration.

    """
    with open(path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config

def save_config(x, path):
    """
    Save the configuration dict 'x' as a YAML file at the specified 'path'.

    Args:
        x (dict): The configuration dict to be saved.
        path (str): The path where the YAML file should be saved.

    Returns:
        None
    """
    with open(os.path.join(path, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(x, yaml_file, default_flow_style=False, sort_keys=False)

def dataloader(data, num_thread, map_func, batch_size, strict=True, factor=10):
    """
    A function that creates a data loader for processing data in parallel.

    Args:
        data: the data to be processed, which can be a list, tuple, or DataFlow object.
        num_thread: The number of threads to use for parallel processing.
        map_func: The function to apply to each data item.
        batch_size: the returned batch size.
        strict: 

    Returns:
        A data loader that fetch the input data in parallel.

    """
    from mitax.dataflow.common import BatchData
    from mitax.dataflow.parallel_map import MultiThreadMapData

    d1 = MultiThreadMapData(data, num_thread, map_func, buffer_size=batch_size*factor, strict=strict)
    return BatchData(d1, batch_size, use_list=False)


def fileflow(files, shuffle=False):
    """
    A data flow class for iterating over a list of files.

    Args:
        files (list): List of file names.
        shuffle (bool): Whether to shuffle the file names.

    Returns:
        Iterator: An iterator that yields file names.
    """
    
    from mitax.dataflow.parallel_map import fileflow as fileflow_

    return fileflow_(files, shuffle)


def create_folder(save_path, time=True, prefix=None):
    """
    Create a folder for logs.

    Parameters:
    save_path (str): The path where the folder will be created.
    time (bool, optional): Whether to include the current timestamp in the folder name. 
                           Defaults to True.

    Returns:
    str: The path of the created folder.
    """
    if time:
        if prefix is not None:
            log_path = os.path.join(save_path, "%s-%s"%(prefix, datetime.now().strftime("%Y%m%d-%H%M%S")))
        else:
            log_path = os.path.join(save_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        log_path = save_path

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path

def list_files(path, ext=None, sort=True):
    """
    List all files in a directory.

    Args:
        path (str): The path to the directory.
        ext (str, optional): The extension of the files to be listed. Defaults to None.
        sort (bool, optional): Whether to sort the files. Defaults to True.

    Returns:
        list: A list of file names.
    """
    if ext is None:
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(ext)]
    if sort:
        files.sort()
    return files

def read_filelist(filename):
    """
    Read a file containing a list of file names.
    """
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
        return lines

def check_out(cmd, split=True):
    """ utility to check_out terminal command and return the output"""

    strs = subprocess.check_output(cmd, shell=True).decode()

    if split:
        split_strs = strs.split('\n')[:-1]
    return split_strs

_RNG_SEED = None

def fix_rng_seed(seed):
    """
    Call this function at the beginning of program to fix rng seed within tensorpack.

    Args:
        seed (int):

    Note:
        See https://github.com/tensorpack/tensorpack/issues/196.

    Example:

        Fix random seed in both tensorpack and tensorflow.

    .. code-block:: python

            seed = 42
            utils.fix_rng_seed(seed)
            tesnorflow.set_random_seed(seed)
            # run trainer
    """
    global _RNG_SEED
    _RNG_SEED = int(seed)

def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)

##################################################
#   Utility Functions for MRI files and recon    #
##################################################

def readcfl(name):
    """
    Read a cfl file and return the data as a NumPy array.

    Parameters:
    name (str): The name of the cfl file (without the extension).

    Returns:
    numpy.ndarray: The data stored in the cfl file, reshaped according to the dimensions specified in the corresponding .hdr file.

    """
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    return a.reshape(dims, order='F')

def writecfl(name, array):
    """
    Write a NumPy array to a file in the .cfl format.

    Parameters:
    name (str): The base name of the output file.
    array (ndarray): The NumPy array to be written.

    Returns:
    None
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
        h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()

def writemulticfl(name, arrays):
    size = 0
    dims = []

    for array in arrays:
        size += array.size
        dims.append(array.shape)

    with open(name + ".hdr", "wt") as h:
        h.write('# Dimensions\n')
        h.write("%d\n" % size)

        h.write('# SizesDimensions\n')
        for dim in dims:
            h.write("%d " % len(dim))
        h.write('\n')

        h.write('# MultiDimensions\n')
        for dim in dims:
            for i in dim:
                h.write("%d " % i)
            h.write('\n')
            
    size = size * np.dtype(np.complex64).itemsize

    with open(name + ".cfl", "a+b") as d:
        os.ftruncate(d.fileno(), size)
        mm = mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE)
        for array in arrays:
            if array.dtype != np.complex64:
                array = array.astype(np.complex64)
            mm.write(np.ascontiguousarray(array.T))
        mm.close()

def bart(nargout, cmd, *args, return_str=False):
    """
    Call bart from the system command line.

    Args:
        nargout (int): The number of output arguments expected from the command.
        cmd (str): The command to be executed by bart.
        *args: Variable number of input arguments for the command.
        return_str (bool, optional): Whether to return the output as a string. Defaults to False.

    Returns:
        list or str: The output of the command. If nargout is 1, returns a single element list.
                     If return_str is True, returns the output as a string.

    Raises:
        Exception: If the command exits with an error.

    Usage:
        bart(<nargout>, <command>, <arguments...>)
    """
    if type(nargout) != int or nargout < 0:
        print("Usage: bart(<nargout>, <command>, <arguments...>)")
        return None

    name = tempfile.NamedTemporaryFile().name

    nargin = len(args)
    infiles = [name + 'in' + str(idx) for idx in range(nargin)]
    in_str = ' '.join(infiles)

    for idx in range(nargin):
        writecfl(infiles[idx], args[idx])

    outfiles = [name + 'out' + str(idx) for idx in range(nargout)]
    out_str = ' '.join(outfiles)

    shell_str = 'bart ' + cmd + ' ' + in_str + ' ' + out_str
    print(shell_str)
    if not return_str:
        ERR = os.system(shell_str)
    else:
        try:
            strs = subprocess.check_output(shell_str, shell=True).decode()
            return strs
        except:
            ERR = True

    for elm in infiles:
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    output = []
    for idx in range(nargout):
        elm = outfiles[idx]
        if not ERR:
            output.append(readcfl(elm))
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    if ERR:
        print("Make sure you install bart properly")
        raise Exception("Command exited with an error.")

    if nargout == 1:
        output = output[0]

    return output

def float2cplx(float_in):
    if isinstance(float_in, np.ndarray):
        return np.array(float_in[...,0]+1.0j*float_in[...,1], dtype='complex64')
    elif isinstance(float_in, jnp.ndarray):
        return jnp.array(float_in[...,0]+1.0j*float_in[...,1], dtype='complex64')
    else:
        raise ValueError('Input must be numpy or jax array')

def cplx2float(cplx_in):
    if isinstance(cplx_in, np.ndarray):
        return np.array(np.stack((cplx_in.real, cplx_in.imag), axis=-1), dtype='float32')
    elif isinstance(cplx_in, jnp.ndarray):
        return jnp.array(jnp.stack((cplx_in.real, cplx_in.imag), axis=-1), dtype='float32')
    else:
        raise ValueError('Input must be numpy or jax array')

def norm_to_uint(inp, bit=8):
    maximum = np.max(inp)
    out = inp/maximum*np.power(2., bit)
    
    tp = np.uint8
    if bit == 16:
        tp= np.uint16
    if bit == 32:
        tp= np.uint16
    return out.astype(tp)

def psnr(img1, img2, bit=16, tobit=False):
    """
    calculate peak SNR, img1-true, img2-test
    """
    pixel_max = np.max(img2)
    
    if tobit:
        img1 = norm_to_uint(img1, bit)
        img2 = norm_to_uint(img2, bit)
        pixel_max = np.power(2., bit) - 1
    return peak_signal_noise_ratio(img1, img2, data_range=pixel_max)

def ssim(img1, img2, bit=16, tobit=False):
    """
    calcualte similarity index between img1 and img2
    """
    
    scale = np.max(img2)
    if tobit:
        img1 = norm_to_uint(img1, bit)
        img2 = norm_to_uint(img2, bit)  
        scale = np.power(2., bit) - 1 
    img1.astype(np.float64)
    img2.astype(np.float64)
    scale.astype(np.float64)
    return structural_similarity(img1, img2, data_range=scale)

##################################################
#   Utility Functions for JAX and numpy          #
##################################################

def batch_add(a, b):
  return jax.vmap(lambda a, b: a + b)(a, b)

def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)


##################################################
#   Utility Functions for create sampler         #
##################################################

def get_last_folder(directory, prefix):
    """
    Get the name of the last folder matching the specified prefix and number pattern
    in the specified directory.

    Args:
    - directory (str): The path to the directory containing folders.
    - prefix (str): The prefix string common to all folders.

    Returns:
    - str: The name of the last folder matching the pattern.
           Returns None if no matching folders are found.
    """
    # Construct the regular expression pattern
    pattern = re.compile(fr'{re.escape(prefix)}_\d+')

    # Get a list of folders matching the pattern
    folders = [folder for folder in os.listdir(directory) if pattern.fullmatch(folder)]

    # If there are folders found
    if folders:
        # Extract numbers from folder names and get the maximum
        last_folder = max(map(lambda x: int(re.search(r'\d+', x).group()), folders))
        return f"{prefix}_{last_folder}"
    else:
        return None
    
def create_sampler(sampler_cls, init_input, logdir):
    config = load_config(logdir+'/config.yaml')
    path = get_last_folder(logdir, config['net_name'])
    print(f'create sampler with the model from {path}')

    return get_class_by_name(sampler, sampler_cls)(net_name  = config['net_name'],
                        net_hparams   = config['net_hparams'],
                        dm_name       = config['loss_name'],
                        dm_hparams    = config['loss_params'],
                        init_input    = init_input,
                        path          = os.path.join(logdir, path))