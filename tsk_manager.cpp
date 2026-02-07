#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <iostream>
#include <vector>
#include <limits>

struct Pose {
    double x; double y; double w;
};

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

class StableTaskManager {
private:
    double penalty_time = 0.0;
    double reward_time = 0.0;

    // 预设坐标点 (成员 B 需根据成员 A 的地图实测修改)
    Pose yellow_zones[4] = {{1.5, 1.5, 1.0}, {-1.5, 1.5, 1.0}, {-1.5, -1.5, 1.0}, {1.5, -1.5, 1.0}};
    Pose pink_zones[4] = {{2.2, 2.2, 1.0}, {-2.2, 2.2, 1.0}, {-2.2, -2.2, 1.0}, {2.2, -2.2, 1.0}};

public:
    // 带有超时和异常处理的导航函数
    bool safeMoveTo(MoveBaseClient &ac, Pose p, std::string label, double timeout_sec = 60.0) {
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose.header.frame_id = "map";
        goal.target_pose.header.stamp = ros::Time::now();
        goal.target_pose.pose.position.x = p.x;
        goal.target_pose.pose.position.y = p.y;
        goal.target_pose.pose.orientation.w = p.w;

        ROS_INFO("任务：前往 %s (%.2f, %.2f)", label.c_str(), p.x, p.y);
        ac.sendGoal(goal);

        // 异常处理 1：导航超时监控
        bool finished = ac.waitForResult(ros::Duration(timeout_sec));
        if (!finished) {
            ROS_ERROR("错误：%s 导航超时！取消当前目标。", label.c_str());
            ac.cancelGoal();
            return false;
        }

        // 异常处理 2：状态检查
        if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED) {
            ROS_INFO("成功：到达 %s", label.c_str());
            return true;
        } else {
            ROS_ERROR("失败：无法到达 %s，状态：%s", label.c_str(), ac.getState().toString().c_str());
            return false;
        }
    }

    // 处理房间任务与分值逻辑
    void runRoomLogic(MoveBaseClient &ac, int id) {
        if (!safeMoveTo(ac, yellow_zones[id], "房间" + std::to_string(id) + "黄色答题区")) {
            ROS_WARN("导航异常，跳过识别，直接判定为失败绕行。");
            safeMoveTo(ac, pink_zones[id], "强制绕行点");
            return;
        }

        int choice = 0, correct = 0;
        // 异常处理 3：健全的用户输入逻辑
        while (true) {
            std::cout << "\n[房间 " << id << "] 输入目标数(1/2) & 正确数(0/1/2): ";
            if (std::cin >> choice >> correct && choice >= 1 && choice <= 2 && correct >= 0 && correct <= choice) {
                break;
            } else {
                std::cout << "输入无效！请确保 正确数 <= 目标数。" << std::endl;
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
        }

        // 计分逻辑
        if (choice == 1 && correct == 1) penalty_time += 10.0;
        else if (choice == 2) {
            if (correct == 1) penalty_time += 5.0;
            else if (correct == 2) reward_time += 10.0;
        }

        // 异常处理 4：强制绕行逻辑
        if (correct == 0) {
            ROS_WARN("识别正确数为0，启动强制绕行。");
            safeMoveTo(ac, pink_zones[id], "粉色区域");
        }
    }

    void report(double duration) {
        ROS_INFO("------------------------------");
        ROS_INFO("原始时间: %.2f s | 罚时: +%.2f s | 奖励: -%.2f s", duration, penalty_time, reward_time);
        ROS_INFO("最终成绩: %.2f s", duration + penalty_time - reward_time);
        ROS_INFO("------------------------------");
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "task_manager_node");
    StableTaskManager manager;
    MoveBaseClient ac("move_base", true);

    if (!ac.waitForServer(ros::Duration(10.0))) {
        ROS_FATAL("无法连接到 move_base 服务器，请检查导航节点是否启动！");
        return 1;
    }

    std::vector<int> order(4);
    std::cout << "请输入房间顺序 (0 1 2 3 范围): ";
    for (int i = 0; i < 4; ++i) {
        while (!(std::cin >> order[i] && order[i] >= 0 && order[i] <= 3)) {
            std::cout << "输入错误，请输入 0-3：";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }

    ros::Time start = ros::Time::now();
    for (int id : order) manager.runRoomLogic(ac, id);
    manager.report((ros::Time::now() - start).toSec());

    return 0;
}
