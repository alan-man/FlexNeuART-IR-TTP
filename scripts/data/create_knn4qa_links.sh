#!/bin/bash
function check {
  f="$?"
  name=$1
  if [ "$f" != "0" ] ; then
    echo "**************************************"
    echo "* Failed: $name"
    echo "**************************************"
    exit 1
  fi
}
DATA_ROOT=$1
if [ "$DATA_ROOT" = "" ] ; then
  echo "Specify the data directory root (1st arg)!"
  exit 1
fi
if [ ! -d "$DATA_ROOT" ] ; then
  echo "'$DATA_ROOT' is not a directory (1st arg)!"
  exit 1
fi
#lucene_index                                
ln -s "/tempory/the_three_potatoes/ri_project/workspaces/lucene_index"
check "ln -s "/tempory/the_three_potatoes/ri_project/workspaces/lucene_index""
#input                                
ln -s "$DATA_ROOT/input"
check "ln -s "$DATA_ROOT/input""
#memfwdindex                          
ln -s "/tempory/the_three_potatoes/ri_project/workspaces/memfwdindex"
check "ln -s "/tempory/the_three_potatoes/ri_project/workspaces/memfwdindex""
#output                               
ln -s "/tempory/the_three_potatoes/ri_project/workspaces/output/"
check "ln -s "/tempory/the_three_potatoes/ri_project/workspaces/output/""
#WordEmbeddings                       
ln -s "$DATA_ROOT/WordEmbeddings/"
check "ln -s "$DATA_ROOT/WordEmbeddings/""
#tran
mkdir -p tran
check "mkdir -p tran"
cd tran
check "cd tran"
ln -s "$DATA_ROOT/tran/ComprMinusManner" manner
check "ln -s "$DATA_ROOT/tran/ComprMinusManner" manner"
ln -s "$DATA_ROOT/tran/ComprMinusManner" ComprMinusManner
check "ln -s "$DATA_ROOT/tran/ComprMinusManner" ComprMinusManner"
ln -s "$DATA_ROOT/tran/compr" compr
check "ln -s "$DATA_ROOT/tran/compr" compr"
ln -s "$DATA_ROOT/tran/stackoverflow" stackoverflow
check "ln -s "$DATA_ROOT/tran/compr" stackoverflow"
