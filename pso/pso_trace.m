X=x';
Y=y';
Z=all_fitness;

K=zeros(2500,3);

for i=1:1:50
    for j=1:1:50
        K(((i-1)*50+j),1)=X(i,j);
        K(((i-1)*50+j),2)=Y(i,j);
        K(((i-1)*50+j),3)=Z(i,j);
    end
end

% 假设你的 N x 3 矩阵名为 pointsMatrix

% 分离 X, Y, Z 坐标到三个单独的向量
X = K(:,1);
Y = K(:,2);
Z = K(:,3);

% 定义新的网格范围和分辨率
xqmin = min(X); xqmax = max(X); % X 轴范围
yqmin = min(Y); yqmax = max(Y); % Y 轴范围
npts = 1000; % 网格点的数量（可以调整）

% 创建查询点的网格
[xq, yq] = meshgrid(linspace(xqmin, xqmax, npts), linspace(yqmin, yqmax, npts));

% 使用 scatteredInterpolant 插值以获取网格上的 Z 值
F = scatteredInterpolant(X, Y, Z);
zq = F(xq, yq);

% 创建一个新的图形窗口
figure;

% 使用 contourf 函数绘制填充的等高线图
contourf(xq, yq, zq, 10); % 20 表示等高线的数量

% 添加颜色条以显示值与颜色的关系
colorbar;

% 添加标题和轴标签
xlabel('Length (nm)');
ylabel('Diameter (nm)');

% 可选：调整颜色映射
colormap parula; % 更改颜色方案为 "jet"

% 可选：添加网格线
grid on;