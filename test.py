import numpy as np
from tsaug import TimeWarp
import matplotlib.pyplot as plt


def create_test_signal(length=100):
    """
    生成测试用的正弦波信号

    Parameters:
    -----------
    length : int
        信号长度

    Returns:
    --------
    numpy.ndarray
        形状为 (1, length, 1) 的测试数据
    """
    x = np.linspace(0, 4 * np.pi, length)
    y = np.sin(x)
    return y.reshape(1, -1, 1)


def test_seed_behavior(X):
    """
    测试1: 比较相同种子和不同种子的行为
    """
    print("\n" + "=" * 60)
    print("测试1: 相同种子 vs 不同种子")
    print("=" * 60)

    # 场景1: 两次都设置相同的种子
    np.random.seed(42)
    aug1 = TimeWarp(n_speed_change=2, max_speed_ratio=3)
    X_aug_1a = aug1.augment(X)

    np.random.seed(42)  # 再次设置相同的种子
    aug2 = TimeWarp(n_speed_change=2, max_speed_ratio=3)
    X_aug_1b = aug2.augment(X)

    print("\n【场景1】两次都设置 seed=42:")
    print(f"第一次增强结果 (前10个值): {X_aug_1a[0, :10, 0].round(3)}")
    print(f"第二次增强结果 (前10个值): {X_aug_1b[0, :10, 0].round(3)}")
    print(f"两次结果是否完全相同? {np.array_equal(X_aug_1a, X_aug_1b)}")

    # 场景2: 只设置一次种子
    np.random.seed(42)
    aug3 = TimeWarp(n_speed_change=2, max_speed_ratio=3)
    X_aug_2a = aug3.augment(X)

    # 注意：这里没有重新设置种子
    aug4 = TimeWarp(n_speed_change=2, max_speed_ratio=3)
    X_aug_2b = aug4.augment(X)

    print("\n【场景2】只设置一次 seed=42:")
    print(f"第一次增强结果 (前10个值): {X_aug_2a[0, :10, 0].round(3)}")
    print(f"第二次增强结果 (前10个值): {X_aug_2b[0, :10, 0].round(3)}")
    print(f"两次结果是否完全相同? {np.array_equal(X_aug_2a, X_aug_2b)}")

    return X_aug_1a, X_aug_1b, X_aug_2a, X_aug_2b


def test_multiply_operator(X):
    """
    测试2: 验证 * 操作符的行为
    """
    print("\n" + "=" * 60)
    print("测试2: 验证 * 操作符的行为")
    print("=" * 60)

    # 测试 * 操作符：一次生成多个版本
    np.random.seed(42)
    n_versions = 3
    aug5 = TimeWarp(n_speed_change=2, max_speed_ratio=3) * n_versions
    X_aug_multiple = aug5.augment(X)

    print(f"\n使用 TimeWarp() * {n_versions} 生成的数据形状: {X_aug_multiple.shape}")
    print(f"生成的{n_versions}个版本是否互不相同?")
    for i in range(n_versions):
        for j in range(i + 1, n_versions):
            is_same = np.array_equal(X_aug_multiple[i], X_aug_multiple[j])
            print(f"  版本{i + 1} vs 版本{j + 1}: {'相同' if is_same else '不同'}")

    return X_aug_multiple


def test_random_generator_state():
    """
    测试3: 验证随机数生成器的状态
    """
    print("\n" + "=" * 60)
    print("测试3: 验证随机数生成器的状态")
    print("=" * 60)

    # 演示随机数生成器的状态
    np.random.seed(42)
    print(f"设置 seed=42 后的第一个随机数: {np.random.rand():.6f}")
    print(f"第二个随机数: {np.random.rand():.6f}")
    print(f"第三个随机数: {np.random.rand():.6f}")

    np.random.seed(42)  # 重置
    print(f"\n重置 seed=42 后的第一个随机数: {np.random.rand():.6f} (与上面第一个相同)")
    print(f"第二个随机数: {np.random.rand():.6f} (与上面第二个相同)")

    # 不重置
    print(f"\n继续（不重置）的第一个随机数: {np.random.rand():.6f} (新的)")
    print(f"继续的第二个随机数: {np.random.rand():.6f} (新的)")


def plot_results(X, X_aug_1a, X_aug_1b, X_aug_2a, X_aug_2b, X_aug_multiple):
    """
    测试4: 可视化对比所有结果
    """
    print("\n" + "=" * 60)
    print("测试4: 可视化对比")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    time_axis = np.arange(X.shape[1])

    # 原始信号
    axes[0, 0].plot(time_axis, X[0, :, 0], 'b-', linewidth=2)
    axes[0, 0].set_title('原始信号')
    axes[0, 0].grid(True, alpha=0.3)

    # 场景1的两个结果（相同种子）
    axes[0, 1].plot(time_axis, X_aug_1a[0, :, 0], 'r-', alpha=0.7, label='第一次')
    axes[0, 1].plot(time_axis, X_aug_1b[0, :, 0], 'g--', alpha=0.7, label='第二次')
    axes[0, 1].set_title('场景1: 两次都设置seed=42')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 场景1的差异
    axes[0, 2].plot(time_axis, X_aug_1a[0, :, 0] - X_aug_1b[0, :, 0], 'k-')
    axes[0, 2].set_title('场景1的差异（接近0）')
    axes[0, 2].grid(True, alpha=0.3)

    # 场景2的两个结果（只设置一次种子）
    axes[1, 0].plot(time_axis, X_aug_2a[0, :, 0], 'r-', alpha=0.7, label='第一次')
    axes[1, 0].plot(time_axis, X_aug_2b[0, :, 0], 'g--', alpha=0.7, label='第二次')
    axes[1, 0].set_title('场景2: 只设置一次seed=42')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 场景2的差异
    axes[1, 1].plot(time_axis, X_aug_2a[0, :, 0] - X_aug_2b[0, :, 0], 'k-')
    axes[1, 1].set_title('场景2的差异（非零）')
    axes[1, 1].grid(True, alpha=0.3)

    # *3生成的三个版本
    for i in range(X_aug_multiple.shape[0]):
        axes[1, 2].plot(time_axis, X_aug_multiple[i, :, 0], alpha=0.5, label=f'版本{i + 1}')
    axes[1, 2].set_title('TimeWarp() * 3 生成的3个版本')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n可视化完成！请查看生成的图表。")


def simplified_test():
    """
    精简版测试：快速验证，不包含可视化
    """
    print("\n" + "=" * 50)
    print("精简版测试（快速验证）")
    print("=" * 50)

    # 生成测试数据
    X = np.sin(np.linspace(0, 4 * np.pi, 50)).reshape(1, -1, 1)

    print("\n测试: 两次都设置相同的种子")
    np.random.seed(42)
    r1 = TimeWarp().augment(X)

    np.random.seed(42)
    r2 = TimeWarp().augment(X)

    print(f"完全相同: {np.array_equal(r1, r2)}")
    print(f"r1前5个值: {r1[0, :5, 0].round(3)}")
    print(f"r2前5个值: {r2[0, :5, 0].round(3)}")

    print("\n测试: 只设置一次种子")
    np.random.seed(42)
    r3 = TimeWarp().augment(X)
    r4 = TimeWarp().augment(X)  # 不重置种子

    print(f"完全相同: {np.array_equal(r3, r4)}")
    print(f"r3前5个值: {r3[0, :5, 0].round(3)}")
    print(f"r4前5个值: {r4[0, :5, 0].round(3)}")


def main():
    """
    主函数：运行所有测试
    """
    print("=" * 60)
    print("TimeWarp 随机性行为测试")
    print("=" * 60)

    # 生成测试数据
    X = create_test_signal(length=100)
    print(f"原始数据形状: {X.shape}")
    print(f"原始数据前10个值: {X[0, :10, 0].round(3)}")

    # 运行测试1：种子行为测试
    X_aug_1a, X_aug_1b, X_aug_2a, X_aug_2b = test_seed_behavior(X)

    # 运行测试2：*操作符测试
    X_aug_multiple = test_multiply_operator(X)

    # 运行测试3：随机数生成器状态测试
    test_random_generator_state()

    # 运行测试4：可视化（可选）
    try:
        plot_results(X, X_aug_1a, X_aug_1b, X_aug_2a, X_aug_2b, X_aug_multiple)
    except Exception as e:
        print(f"\n可视化失败（可能是缺少matplotlib）：{e}")
        print("运行精简版测试替代...")
        simplified_test()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()