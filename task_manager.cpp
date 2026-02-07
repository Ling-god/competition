#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <iostream>
#include <vector>

// 定义坐标结构体
struct Pose {
    double x;
    double y;
    double w; // 四元数 z 轴分量，简化处理
};

// 定义 Actionlib 客户端类型
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

class CompetitionTask {
private:
    double total_penalty = 0.0; // 累计罚时
    double reward_bonus = 0.0;  // 累计奖励扣除 (双目标全对减10s)
    
    // 房间坐标定义 (由成员 A 扫描地图后提供)
    // 假设坐标顺序对应房间 0, 1, 2, 3
    Pose yellow_zones[4] = { {1.2, 1.5, 1.0}, {-1.5, 1.3, 1.0}, {-1.3, -1.4, 1.0}, {1.4, -1.6, 1.0} };
    Pose pink_zones[4] = { {2.0, 2.0, 1.0}, {-2.0, 2.0, 1.0}, {-2.0, -2.0, 1.0}, {2.0, -2.0, 1.0} };

public:
    // 发送目标点函数
    bool goTo(MoveBaseClient &ac, Pose target) {
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose.header.frame_id = "map";
        goal.target_pose.header.stamp = ros::Time::now();
        goal.target_pose.pose.position.x = target.x;
        goal.target_pose.pose.position.y = target.y;
        goal.target_pose.pose.orientation.w = target.w;

        ROS_INFO("正在前往坐标: (%.2f, %.2f)", target.x, target.y);
        ac.sendGoal(goal);
        ac.waitForResult(); // 阻塞等待机器人到达

        if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED) {
            ROS_INFO("到达目标点！");
            return true;
        } else {
            ROS_WARN("导航失败或被放弃！");
            return false;
        }
    }

    // 处理识别逻辑与奖惩
    void processRoomTask(MoveBaseClient &ac, int room_id) {
        int target_choice, correct_count;
        
        // 1. 到达黄色答题区
        goTo(ac, yellow_zones[room_id]);
        
        // 2. 模拟获取识别结果 (实际比赛中此处调用视觉接口)
        std::cout << "房间 " << room_id << " 识别结果：\n";
        std::cout << "  你选择识别几个目标 (1 或 2): "; std::cin >> target_choice;
        std::cout << "  识别正确了几个 (0/1/2): "; std::cin >> correct_count;

        // 3. 奖惩判定逻辑
        if (target_choice == 1) {
            if (correct_count == 1) total_penalty += 10.0; // 选单目标且正确罚10s
        } else if (target_choice == 2) {
            if (correct_count == 1) total_penalty += 5.0;  // 选双目标对1个罚5s
            if (correct_count == 2) reward_bonus += 10.0;  // 选双目标全对扣除10s奖励
        }

        // 4. 强制绕行逻辑：若正确数为 0，必须去粉色区域
        if (correct_count == 0) {
            ROS_INFO("识别正确数为0，执行强制绕行粉色区域逻辑...");
            goTo(ac, pink_zones[room_id]);
        }
    }

    void finalize(double total_time) {
        double final_score = total_time + total_penalty - reward_bonus;
        ROS_INFO("======= 比赛结算 =======");
        ROS_INFO("原始用时: %.2f 秒", total_time);
        ROS_INFO("累计罚时: %.2f 秒", total_penalty);
        ROS_INFO("奖励扣除: %.2f 秒", reward_bonus);
        ROS_INFO("最终成绩: %.2f 秒", final_score);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "task_manager_node");
    CompetitionTask task;
    MoveBaseClient ac("move_base", true);

    // 等待导航服务器启动
    while(!ac.waitForServer(ros::Duration(5.0))) {
        ROS_INFO("等待 move_base 服务器...");
    }

    // 接收裁判输入的房间顺序 (例如输入 4 个数字)
    std::vector<int> room_order(4);
    std::cout << "请输入房间访问顺序 (0-3，空格分隔): ";
    for(int i=0; i<4; i++) std::cin >> room_order[i];

    ros::Time start_time = ros::Time::now();

    // 循环访问每个房间
    for(int id : room_order) {
        task.processRoomTask(ac, id);
    }

    // 所有任务结束，计算成绩
    double duration = (ros::Time::now() - start_time).toSec();
    task.finalize(duration);

    return 0;
}
