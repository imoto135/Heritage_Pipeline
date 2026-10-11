function matlab_reference_export()
% Run with MATLAB, Image Processing Toolbox, and a configured MEX compiler.
% Add this file's directory to MATLAB's path before calling the function.
base = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(base)));
src = fullfile(repo, 'tmp_work', 'classical_deep_review_20261008', 'sources', 'howe_ijdar');
maxsrc = fullfile(repo, 'tmp_work', 'classical_deep_review_20261008', 'sources', 'maxflow-v3.01.src');
addpath(src);
dest = fullfile(base, 'matlab_reference');
if ~exist(dest, 'dir'), mkdir(dest); end
if ~exist(fullfile(dest, ['imgcut3.' mexext]), 'file')
  mex('-outdir', dest, '-output', 'imgcut3', ['-I' src], ['-I' maxsrc], ...
      fullfile(src, 'imgcut3.cpp'), fullfile(maxsrc, 'graph.cpp'), fullfile(maxsrc, 'maxflow.cpp'));
end
if ~exist(fullfile(dest, ['imgcutmulti.' mexext]), 'file')
  mex('-outdir', dest, '-output', 'imgcutmulti', ['-I' src], ['-I' maxsrc], ...
      fullfile(src, 'imgcutmulti.cpp'), fullfile(maxsrc, 'graph.cpp'), fullfile(maxsrc, 'maxflow.cpp'));
end
addpath(dest);
names = {'200010454_00002_2', 'hnsd007_042', 'hnsd001_011'};
paths = {
  fullfile(repo, 'data', 'split_dataset', 'test', '200010454', '200010454_00002_2.jpg'), ...
  fullfile(repo, 'data', 'split_dataset', 'train', 'hnsd00000', 'hnsd007_042.jpg'), ...
  fullfile(repo, 'data', 'split_dataset', 'train', 'hnsd00000', 'hnsd001_011.jpg')};
for i = 1:numel(names)
  raw = imread(paths{i});
  if size(raw,3)==3, gray=rgb2gray(raw); else, gray=raw; end
  imwrite(gray, fullfile(dest, [names{i} '_gray.png']));
  edges = edge(gray, 'canny_old', [0.1 0.4], 0.6);
  imwrite(edges, fullfile(dest, [names{i} '_edges.png']));
  fixed = binarizeImage(raw, 160, 0.4, 0.1, 0.6);
  imwrite(uint8(~fixed)*255, fullfile(dest, [names{i} '_fixed.png']));
  [auto, c, thi] = binarizeImageAlg3(raw, 0.1, 0.6);
  imwrite(uint8(~auto)*255, fullfile(dest, [names{i} '_auto.png']));
  fid=fopen(fullfile(dest, [names{i} '_auto_parameters.txt']), 'w');
  fprintf(fid, 'c=%.17g\nthi=%.17g\n', c, thi); fclose(fid);
end
end
