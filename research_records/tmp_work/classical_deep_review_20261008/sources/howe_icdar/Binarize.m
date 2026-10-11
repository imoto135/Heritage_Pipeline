% Call this function with the file name of the source image and the
% file name for the output image.

function Binarize(infile,outfile)

img = imread(infile);
bimg = binarizeImage(img);
imwrite(bimg,outfile);
end


function bimg = binarizeImage(img,varargin)

if (nargin < 2)|isempty(varargin{1})
    wgt = 122.1634;
else
    wgt = varargin{1};
end;
if (nargin < 3)|isempty(varargin{2})
    sr = 23.2084;
else
    sr = varargin{2};
end;
if (nargin < 4)|isempty(varargin{3})
    ct = [5.2742e-004 0.3899];
else
    ct = varargin{3};
end;
if (size(img,3)==3)
    img = rgb2gray(img);
end;
eimg = edge(img,'canny',ct);
if isa(img,'uint8')
    img = double(img);
else
    img = 255.*img; % numbers are set for 255 scale
end;

dx = img(1:end-1,2:end)-img(1:end-1,1:end-1);
dy = img(2:end,1:end-1)-img(1:end-1,1:end-1);
dvr = divergence(dx,dy);

img2 = (img-gsmooth(img,sr,3*sr,'mirror'));
rms = sqrt(gsmooth(img2.^2,sr));
himask = (img2./(rms+eps)>2);
dvr(himask(1:end-1,1:end-1)) = -500;

hc = ~((eimg(1:end-1,1:end-1)&(dy>0))|(eimg(2:end,1:end-1)&(dy<=0)));
vc = ~((eimg(1:end-1,1:end-1)&(dx>0))|(eimg(1:end-1,2:end)&(dx<=0)));
hc = hc(1:end-1,:);
vc = vc(:,1:end-1);

bimg = logical(imgcut3(1500-dvr,1500+dvr,wgt.*hc,wgt.*vc));
bimg = bimg([1:end end],[1:end end]);

% refine edge map by adding faint edges within detected foreground
eimg2 = edge(img,'canny',0);
eimg = eimg|(eimg2&bimg);

hc = ~((eimg(1:end-1,1:end-1)&(dy>0))|(eimg(2:end,1:end-1)&(dy<=0)));
vc = ~((eimg(1:end-1,1:end-1)&(dx>0))|(eimg(1:end-1,2:end)&(dx<=0)));
hc = hc(1:end-1,:);
vc = vc(:,1:end-1);

% redo binarization
bimg = logical(imgcut3(1500-dvr,1500+dvr,wgt.*hc,wgt.*vc));
bimg = bimg([1:end end],[1:end end]);
%nw = eimg&~eimg([1 1:end-1],:)&~eimg(:,[1 1:end-1]);
nw = bimg&~bimg([1 1:end-1],:)&~bimg(:,[1 1:end-1]);
bimg = bimg&~nw;
bimg = ~bimg;  % match DIBCO sign convention
end


function sm = gsmooth(m,sigma,radius,method)

if (nargin < 2)|isempty(sigma)
    sigma = 1;
end;

if (nargin < 3)|isempty(radius)
  radius = ceil(2.5*sigma);
else
    radius = ceil(radius);
end;

if (nargin < 4)|isempty(method)
  method='none';
end;

if (sigma == 0)
  sm = m;
else
  hcol = fspecial('gaussian',[2*radius+1,1],sigma);
  hrow = fspecial('gaussian',[1,2*radius+1],sigma);

  [nrow,ncol,nclr] = size(m);
  sm = zeros(nrow,ncol);
  for c = 1:nclr
      switch method
          case 'none'
              sm(:,:,c) = conv2(hcol,hrow,m(:,:,c),'same');
          case 'mirror'
              sm(:,:,c) = conv2(hcol,hrow,padarray(m(:,:,c),[radius radius],'symmetric'),'valid');
      end;
  end;
end;
end