This is a reference code implementation to accompany the paper
"Document Binarization via Graph Cut Energy Minimization with Edge
Detection" by Nicholas R. Howe, published in the proceedings of the
International Conference on Document Analysis and Recognition, 2011.
Those using this code or work derived from it should include a
citation to this paper.

As written the program Binarize.m takes two file names as arguments,
one for the input image file and another for the output.  It calls
binarizeImage.m, which processes the image data.

To compile the imgcut mex file you will need additional source and
header files from the graph cut package written by V. Kolmogorov.
These are separately licensed and available for download at
http://pub.ist.ac.at/~vnk/software/maxflow-v3.01.src.tar.gz.
