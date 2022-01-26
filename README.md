# MAMnet
MAMnet is a long read based structural variant caller uses deep learning network.

MAMnet is able to detect and genotype deletions and insertions with fast running speed.


Installation
------------

    #Install from github (requires Python 3.6.* or newer): installs all dependencies except those necessary for read alignment (ngmlr, minimap2, samtools)
    git clone https://github.com/micahvista/MAMnet.git
    cd MAMnet

Dependencies
------------
- *tensorflow>=2.3.0* 
- *pandas*
- *numpy* 
- *pysam* 
- *numba*
- *scipy*



Input
-----

MAMnet takes sorted and indexed alignment files in BAM format as inputs. And MAMnet has been successfully tested on PacBio CLR, PacBio HiFi (CCS) and Oxford Nanopore data and alignment files produced by the read aligners `minimap2 <https://github.com/lh3/minimap2>`_, `pbmm2 <https://github.com/PacificBiosciences/pbmm2/>`_ , `NGMLR <https://github.com/philres/ngmlr>`_, and BWA-MEM.

Output
------

MAMnet produces SV calls in the Variant Call Format (VCF).

Usage
----------------------
    python MAMnet.py -bamfilepath ./HG002_PB_70x_RG_HP10XtrioRTG.bam -workdir ./workdir -outputpath ./variants.vcf -threads 16 -step 50 -includecontig [1,2,3,4]
    
    #-bamfilepath the inputs path of sort and index bam file, the bam file should has MD tag which can be compuated by samtools calmd...
    #-workdir the work path of MAMnet to store temporary data
    #-outputpath the output path of called vcf file
    #-threads the number of threads to use. (default: all available thread)
    #-step data shift size [1-200]. (default: 50)
    #-includecontig the list of contig to preform detection. (default: [], all contig are used)


Changelog
---------


Contact
-------

If you experience any problems or have suggestions please create an issue or a pull request.

Citation
---------


License
-------

The project is licensed under the GNU General Public License.
