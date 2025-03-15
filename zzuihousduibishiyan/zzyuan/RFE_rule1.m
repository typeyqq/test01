function selected_features = RFE_rule1(X, y, n_features_to_select)
    % 设置随机森林参数
    n_trees = 100;
    rf = TreeBagger(n_trees, X, y, 'Method', 'classification');

    % 获取初始特征索引
    all_features = 1:size(X, 2);

    % 初始化选择的特征集合
    selected_features = [];

    % 递归特征消除主循环
    for i = 1:n_features_to_select
        % 计算当前剩余特征的性能
        current_performance = cross_val_performance(rf, X(:, all_features), y);

        % 记录当前最差特征的索引
        worst_feature = -1;
        worst_performance = inf;

        % 逐个移除特征并计算性能
        for j = 1:length(all_features)
            temp_features = setdiff(all_features, all_features(j));
            temp_performance = cross_val_performance(rf, X(:, temp_features), y);

            % 更新最差特征
            if temp_performance < worst_performance
                worst_feature = j;
                worst_performance = temp_performance;
            end
        end

        % 移除最差特征
        all_features = setdiff(all_features, all_features(worst_feature));

        % 添加当前最差特征到已选择特征集合
        selected_features = [selected_features, all_features(worst_feature)];

        % 打印当前迭代的结果（可选）
        disp(['Iteration ', num2str(i), ': Selected features: ', num2str(selected_features)]);
    end
end

function performance = cross_val_performance(model, X, y)
    cv_model = crossval(@classfun, X, y, 'kfold', 5);  % 5折交叉验证
    performance = kfoldLoss(cv_model);
end
