# Automated ILP Scheduler
This scheduler takes a DFG that is represented in edge- list format as input to automatically generate the schedule and produce the minimized results from the schedule.

Setup
  Use a Linux distribution
  Install Git and clone this repo:
    sudo apt update; sudo apt install git
    git clone 
  
  Install Python and the library 'networkx' and 'tabulate':
    sudo apt install python3 
    pip install networkx tabulate
  
  Install GLPK (in the same directory as this repo)  
    Download GLPK source and unzip the file:
      wget http://ftp.gnu.org/gnu/glpk/glpk-4.35.tar.gz; tar -zxvf glpk-4.35.tar.gz
    
    Install libraries for compilation:
      sudo apt-get install build-essential
    
    Enter the unzipped folder and prepare for compilation:
      cd glpk-4.35; ./configure
    
    Compile and install GLPK to your system:
      make
