function [output] = Belief_Degree_Ascending(y,value)
%BELIEF_DEGREE 此处显示有关此函数的摘要y2 = [ 50, 80, 100, 120 ]  86 
%   此处显示详细说明
    output = [0.0 0.0 0.0 0.0 0.0];
    for i =1:1
        if y(5) < value
            output(5) = 1;
            break;
        end
        if y(1) > value
            output(1) = 1;
            break;
        end
        for k = 1:5
            if k <= 4
                if value >= y(k) && value <= y(k+1)
                    output(k) = (y(k+1) - value)/(y(k+1) - y(k));
                    output(k+1) = 1 - output(k);
                    break;
                end
            else
                break;
            end
        end
    end
end
