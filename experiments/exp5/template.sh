#!/bin/bash

export PATH
. env/bin/activate 

cd src
python main.py {args_imported_by_python}
mv *.csv ../
cd ..
