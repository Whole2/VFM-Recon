
def get_scans_tnt(dataset_dir, pred_depth_dir, bound_dir, scene_list_file=None):
    
    
    scene_whitelist = []
    if scene_list_file is not None and os.path.exists(scene_list_file):
        with open(scene_list_file, 'r', encoding='utf-8') as f:
            # 读取所有行，去除空格、换行符，过滤空行
            scene_whitelist = [line.strip() for line in f if line.strip()]
        print(f"从{scene_list_file}读取到{len(scene_whitelist)}个目标场景")
    elif scene_list_file is not None:
        print(f"警告：场景白名单文件{scene_list_file}不存在，将返回所有场景")
        

    scan_dirs = sorted(
        [f for f in glob.glob(os.path.join(dataset_dir, "scans/*")) if os.path.isdir(f)])
    test_scans = []
    for scan_dir in scan_dirs:
        scan_name = os.path.basename(scan_dir)
        
        if scene_whitelist and scan_name not in scene_whitelist:
            continue
        
        # zero_crossings_npzfile = os.path.join(tsdf_dir, f"{scan_name}_zc.npz")
        pose_npyfile = os.path.join(scan_dir, "poses.npy")
        image_dir = os.path.join(scan_dir, "color")

        if not os.path.exists(pose_npyfile):
            n_poses = len(glob.glob(os.path.join(scan_dir, "pose/*.txt")))
            print(pose_npyfile, n_poses)
            poses = np.stack([np.loadtxt(os.path.join(scan_dir, f"pose/{i}.txt")) for i in range(n_poses)])
            np.save(pose_npyfile, poses)

        scan = {
            "scan_dir": scan_dir,
            "scan_name": scan_name,
            # "zero_crossings_npzfile": zero_crossings_npzfile,
            "pose_npyfile": pose_npyfile,
            "image_dir": image_dir,
            "pred_depth_dir": os.path.join(pred_depth_dir, scan_name),
            "bound_dir":bound_dir,
        }
        test_scans.append(scan)

    return test_scans






def get_scans_tartanair(dataset_dir, pred_depth_dir,scene_list_file=None):
    """
    遍历数据集目录，识别场景
    :param dataset_dir: 数据集根目录
    :param scene_list_file: (可选) 包含指定场景名称的txt文件路径
    """
    # 1. 获取所有可能的目录
    scan_dirs = sorted([dir_path for dir_path in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(dir_path)])
    
    # 2. 读取场景白名单 (如果文件存在且不为空)
    target_scenes = set()
    if scene_list_file and os.path.exists(scene_list_file):
        try:
            with open(scene_list_file, 'r') as f:
                # 读取每一行，去除空白符，过滤空行
                target_scenes = {line.strip() for line in f if line.strip()}
            
            if len(target_scenes) > 0:
                print(f"检测到场景列表文件，将仅处理以下 {len(target_scenes)} 个场景: {target_scenes}")
            else:
                print("场景列表文件为空，将处理所有场景。")
        except Exception as e:
            print(f"读取场景列表文件失败: {e}，将处理所有场景。")
    
    test_scans = []
    
    for scan_dir in scan_dirs:
        scan_name = os.path.basename(scan_dir)
        
        # 3. 过滤逻辑：
        # 如果 target_scenes 有内容（即文件存在且不为空），且当前场景不在列表中，则跳过
        if target_scenes and scan_name not in target_scenes:
            continue
        
        # 路径配置
        image_dir = os.path.join(scan_dir, "image_left")
        depth_dir = os.path.join(scan_dir, "depth_left")
        pose_file = os.path.join(scan_dir, "pose_left.txt")

        if not os.path.exists(pose_file):
            continue

        scan = {
            "scan_dir": scan_dir,
            "scan_name": scan_name,
            "image_dir": image_dir,
            "depth_dir": depth_dir,
            "pred_depth_dir": os.path.join(pred_depth_dir, scan_name),
            "pose_file": pose_file,
        }
        test_scans.append(scan)
        
    return test_scans

#for moge2
def get_scans_moge2(dataset_dir, scene_list_file=None):
    """
    遍历数据集目录，识别符合你描述结构的场景
    结构:
    dataset_dir/
       scan_name/
           images/ (*.jpg)
           depth/ (*.png)
           poses.npy
           scales.npy
           biases.npy
           intrinsic.txt
    """
    # 1. 获取所有子目录
    scan_dirs = sorted([dir_path for dir_path in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(dir_path)])
    
    # 2. 读取场景白名单
    target_scenes = set()
    if scene_list_file and os.path.exists(scene_list_file):
        try:
            with open(scene_list_file, 'r') as f:
                target_scenes = {line.strip() for line in f if line.strip()}
            print(f"检测到场景列表，将处理 {len(target_scenes)} 个场景。")
        except Exception as e:
            print(f"读取场景列表失败: {e}，将处理所有场景。")
    
    test_scans = []
    
    for scan_dir in scan_dirs:
        scan_name = os.path.basename(scan_dir)
        
        # 过滤场景
        if target_scenes and scan_name not in target_scenes:
            continue
            
        # 3. 定义关键文件路径
        image_dir = os.path.join(scan_dir, "color")
        depth_dir = os.path.join(scan_dir, "depth")
        pose_file = os.path.join(scan_dir, "poses.npy")
        scale_file = os.path.join(scan_dir, "scales.npy")
        bias_file = os.path.join(scan_dir, "biases.npy")
        intrinsic_file = os.path.join(scan_dir, "intrinsic/intrinsic_color.txt")
        estimate_scale_dir = os.path.join(scan_dir,"estimate_scale.txt")

        # 4. 检查必要文件是否存在，不存在则跳过该场景
        # 这里假设 poses 和 intrinsic 是必须的，scales/biases 如果用于深度加载也是必须的
        # required_files = [image_dir, depth_dir, pose_file, scale_file, bias_file, intrinsic_file]
        # if not all(os.path.exists(p) for p in required_files):
        #     print(f"跳过场景 {scan_name}: 文件缺失")
        #     continue

        scan = {
            "scan_dir": scan_dir,
            "scan_name": scan_name,
            "image_dir": image_dir,
            "depth_dir": depth_dir,
            "pose_file": pose_file,
            "scale_file": scale_file,
            "bias_file": bias_file,
            "intrinsic_file": intrinsic_file,
            "estimate_scale_dir":estimate_scale_dir
        }
        test_scans.append(scan)
        
    return test_scans






def load_scan_TNT(scan, keyframes_file=None):
    scan_dir = scan["scan_dir"]
    
    sort_key = lambda f: int(os.path.basename(f).split(".")[0])
    rgb_imgfiles = sorted(glob.glob(os.path.join(scan["image_dir"], "*.jpg")), key=sort_key)



    actual_rgb_img_height, actual_rgb_img_width, _ = cv2.imread(rgb_imgfiles[0]).shape

    intr_file = os.path.join(scan_dir, "intrinsic/intrinsic_color.txt")
    K_color = torch.from_numpy(np.loadtxt(intr_file)[:3, :3]).float()
    K_color[0] *= TARGET_RGB_IMG_SIZE[1] / actual_rgb_img_width
    K_color[1] *= TARGET_RGB_IMG_SIZE[0] / actual_rgb_img_height

    poses = np.load(scan["pose_npyfile"])
    
    #xmodified W2C TO C2W
    # poses = np.linalg.inv(poses)
    
    good_pose = ~np.any(np.isinf(poses), axis=(1, 2))

    frames = {i+1: {"pose": poses[i]} for i in range(len(poses))}

    for f in rgb_imgfiles:
        i = int(f.split("/")[-1][:-4])
        frames[i]["rgb_imgfile"] = f


    frames = {i: frame for i, frame in frames.items() if good_pose[i-1]}
    
    # kf_idxs=None #xmodified
    kf_idxs = list(frames.keys())
    if keyframes_file is not None:
        with open(keyframes_file, "r") as f:
            kf_idxs = np.array(json.load(f)[scan["scan_name"]])
            print(scan["scan_name"]+":")#
        frames = {i: frames[i] for i in kf_idxs}

    frame_idxs = sorted(frames.keys())
    poses = torch.from_numpy(np.stack([frames[i]["pose"] for i in frame_idxs]))
    rgb_imgfiles = np.array([frames[i]["rgb_imgfile"] for i in frame_idxs])
    

    return (
        rgb_imgfiles,
        poses,
        K_color,
        kf_idxs,
        scan["bound_dir"],
    )
    
    
def load_scan_tartanair(scan):
    scan_dir = scan["scan_dir"]
    keyframes_file = os.path.join(scan_dir, "keyframes_list.txt")
    
    # 1. 读取图像文件列表
    sort_key = lambda f: int(os.path.basename(f).split("_")[0])
    rgb_imgfiles = sorted(glob.glob(os.path.join(scan["image_dir"], "*_left.png")), key=sort_key)
    gt_depth_files = sorted(glob.glob(os.path.join(scan["depth_dir"], "*_left_depth.npy")), key=sort_key)

    # 2. 生成内参 (保持不变)
    actual_h, actual_w = 480, 640 
    K = np.array([
        [320.0, 0, 320.0],
        [0, 320.0, 240.0],
        [0, 0, 1.0]
    ])
    K_color = torch.from_numpy(K).float()
    K_color[0] *= TARGET_RGB_IMG_SIZE[1] / actual_w
    K_color[1] *= TARGET_RGB_IMG_SIZE[0] / actual_h

    # =========================================================
    # 3. 读取 Pose 并转换为 4x4 矩阵 (重点修改区域 !!!)
    # =========================================================
    raw_poses = np.loadtxt(scan["pose_file"]) 
    
    # 定义 NED 到 Camera 的转换矩阵 (和你的 Fusion 代码一致)
    # 作用: [NED X(前), Y(右), Z(下)] -> [Cam Z(前), X(右), Y(下)]
    T_ned2cam = np.array([[0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
    T_ned2cam_inv = np.linalg.inv(T_ned2cam)

    poses_list = []
    for p in raw_poses:
        # A. 先转成标准的 4x4 矩阵 (此时还是 NED 坐标系)
        T_ned = quaternion_to_matrix(p)
        
        # B. 执行基变换: T_cam = T_convert * T_ned * T_convert_inv
        # 这一步将 Pose 从“NED世界里的NED相机”转换成了“标准世界里的标准相机”
        T_cam = T_ned2cam @ T_ned @ T_ned2cam_inv
        
        poses_list.append(T_cam)

    poses = np.stack(poses_list)
    # =========================================================

    num_frames_available = len(rgb_imgfiles)

    # ----------------------------------------------------
    # --- 关键帧提取逻辑 (保持不变) ---
    # ----------------------------------------------------
    kf_indices = []
    if os.path.exists(keyframes_file):
        try:
            kf_ids = np.loadtxt(keyframes_file, dtype=int)
            if kf_ids.ndim == 0: 
                 kf_ids = np.array([kf_ids])
        except Exception as e:
            print(f"读取关键帧文件失败: {e}")
            kf_ids = np.arange(num_frames_available)
        kf_indices = [idx for idx in kf_ids if idx < num_frames_available]
    else:
        print("未找到关键帧列表文件，加载所有可用帧。")
        kf_indices = np.arange(num_frames_available)

    # 4. 使用关键帧索引筛选数据
    kf_indices = np.array(kf_indices, dtype=int)
    
    # 筛选文件列表
    rgb_imgfiles = [rgb_imgfiles[i] for i in kf_indices]
    gt_depth_files = [gt_depth_files[i] for i in kf_indices]
    
    # 筛选位姿
    poses_kf = torch.from_numpy(poses[kf_indices]).float()

    # 5. 返回关键帧数据
    kf_idx = kf_indices 
    
    return rgb_imgfiles, gt_depth_files, poses_kf, K_color, kf_idx

#for moge2
def load_scan_moge2(scan):
    """
    加载单个场景的元数据（文件列表、位姿、参数）
    """
    # 1. 加载 NPY 数据
    poses = np.load(scan["pose_file"])   # Shape: (N, 4, 4)
    scales = np.load(scan["scale_file"]) # Shape: (N,)
    biases = np.load(scan["bias_file"])  # Shape: (N,)
    
    # 2. 加载内参
    # 假设 intrinsic.txt 存储的是 3x3 或 4x4 矩阵文本
    try:
        K_arr = np.loadtxt(scan["intrinsic_file"])
        # 如果是 4x4，取前3x3；如果是 3x3 则直接用
        if K_arr.shape == (4, 4):
            K = torch.from_numpy(K_arr[:3, :3]).float()
        else:
            K = torch.from_numpy(K_arr).float()
    except Exception as e:
        print(f"内参加载失败 {scan['scan_name']}: {e}")
        raise e

    # 3. 获取图片和深度图列表并排序
    # 图片是 jpg, 深度是 png
    rgb_files = sorted(glob.glob(os.path.join(scan["image_dir"], "*.jpg")))
    depth_files = sorted(glob.glob(os.path.join(scan["depth_dir"], "*.png")))
    estimate_scale=np.loadtxt(scan["estimate_scale_dir"])
    
    # 4. 严谨性检查：数量对齐
    # num_frames = len(poses)
    # if not (len(rgb_files) == len(depth_files) == num_frames == len(scales) == len(biases)):
    #     print(f"警告: 场景 {scan['scan_name']} 数据长度不一致!")
    #     print(f"Poses: {len(poses)}, RGB: {len(rgb_files)}, Depth: {len(depth_files)}, Scales: {len(scales)}")
    #     # 强制截断到最小值以防止报错 (或者你可以选择 raise Error)
    #     min_len = min(len(rgb_files), len(depth_files), num_frames, len(scales))
    #     rgb_files = rgb_files[:min_len]
    #     depth_files = depth_files[:min_len]
    #     poses = poses[:min_len]
    #     scales = scales[:min_len]
    #     biases = biases[:min_len]

    return rgb_files, depth_files, poses, scales, biases, K, estimate_scale


def load_scan_moge2_no_optimize(scan):
    """
    加载单个场景的元数据（文件列表、位姿、参数）
    """
    # 1. 加载 NPY 数据
    poses = np.load(scan["pose_file"])   # Shape: (N, 4, 4)
    
    # 2. 加载内参
    # 假设 intrinsic.txt 存储的是 3x3 或 4x4 矩阵文本
    try:
        K_arr = np.loadtxt(scan["intrinsic_file"])
        # 如果是 4x4，取前3x3；如果是 3x3 则直接用
        if K_arr.shape == (4, 4):
            K = torch.from_numpy(K_arr[:3, :3]).float()
        else:
            K = torch.from_numpy(K_arr).float()
    except Exception as e:
        print(f"内参加载失败 {scan['scan_name']}: {e}")
        raise e

    # 3. 获取图片和深度图列表并排序
    # 图片是 jpg, 深度是 png
    rgb_files = sorted(glob.glob(os.path.join(scan["image_dir"], "*.jpg")))
    depth_files = sorted(glob.glob(os.path.join(scan["depth_dir"], "*.png")))
    print(len(depth_files))
    estimate_scale=np.loadtxt(scan["estimate_scale_dir"])
    
    # 4. 严谨性检查：数量对齐
    # num_frames = len(poses)
    # if not (len(rgb_files) == len(depth_files) == num_frames == len(scales) == len(biases)):
    #     print(f"警告: 场景 {scan['scan_name']} 数据长度不一致!")
    #     print(f"Poses: {len(poses)}, RGB: {len(rgb_files)}, Depth: {len(depth_files)}, Scales: {len(scales)}")
    #     # 强制截断到最小值以防止报错 (或者你可以选择 raise Error)
    #     min_len = min(len(rgb_files), len(depth_files), num_frames, len(scales))
    #     rgb_files = rgb_files[:min_len]
    #     depth_files = depth_files[:min_len]
    #     poses = poses[:min_len]
    #     scales = scales[:min_len]
    #     biases = biases[:min_len]

    return rgb_files, depth_files, poses,  K, estimate_scale






def load_depth_with_scale_and_bias(path, scale, bias,estimate_scale):
    """
    读取PNG深度图并应用 scale 和 bias 转换为真实深度
    """
    # 1. 读取 PNG (读取原始值, flag=-1)
    depth_raw = cv2.imread(path, -1)
    if depth_raw is None:
        raise FileNotFoundError(f"无法读取深度图: {path}")

    # 2. 转为 float32
    depth_val = depth_raw.astype(np.float32)
    
    depth_val/=1000
    
    
    #TODO
    #xmodified 神秘小系数0.4
    # depth_val*=estimate_scale
    
    # 3. 应用公式: depth = depth * scale + bias
    # 注意：这里假设 scale 和 bias 是标量，会自动广播到整个图像
    depth_metric = depth_val * scale + bias
    # depth_metric=depth_val
    
    
    # 4. 过滤负值 (物理深度不能为负)
    depth_metric[depth_metric < 0] = 0
    
    return torch.from_numpy(depth_metric).unsqueeze(0) # (1, H, W)















class TNTDataset(torch.utils.data.Dataset):
    def __init__(self, scans, load_depth=False, keyframes_file=None):
        self.scans = scans
        self.load_depth = load_depth

        self.frames = []
        for scan in self.scans:
            (
                rgb_imgfiles,
                poses,
                K_color,
                kf_idx,
                bound_dir,
            ) = load_scan_TNT(scan, keyframes_file=keyframes_file)
            
            # get bound info
            scene_name = scan["scan_name"]
            origin_path = os.path.join(bound_dir, f"{scene_name}_COLMAP_origin.npy")
            maxbound_path = os.path.join(bound_dir, f"{scene_name}_COLMAP_maxbound.npy")
            gt_origin = torch.from_numpy(np.load(origin_path))
            gt_maxbound = torch.from_numpy(np.load(maxbound_path))
            print(scene_name+" origin and maxbound loaded")
            print(gt_origin)
            print(gt_maxbound)
            
            # #xmodified
            # margin=2
            # pose_mats = torch.tensor(poses).float()                   # (N, 4, 4)
            # cam_centers = pose_mats[:, :3, 3]                         # 所有相机的世界坐标 (N, 3)

            # min_xyz = cam_centers.min(dim=0).values                   # (3,)
            # max_xyz = cam_centers.max(dim=0).values                   # (3,)
            # # 预留余量
            # gt_origin = (min_xyz - margin).clone()
            # gt_maxbound = (max_xyz + margin).clone()
            
            
            # gt_origin[0]=-5
            # gt_maxbound[0]=5
            
            # gt_origin[1]=-5
            # gt_maxbound[1]=5
            
            # gt_origin[2]=-5
            # gt_maxbound[2]=5 
            
                   
            

            for i in range(len(rgb_imgfiles)):
                initial_frame = i == 0
                final_frame = i == len(rgb_imgfiles) - 1
                keyframe = kf_idx[i]

                if load_depth:
                    pred_depth_imgfile = os.path.join(
                        scan["pred_depth_dir"],
                        "depth",
                        os.path.basename(rgb_imgfiles[i])[:-4].zfill(8) + ".png",#xmodified 自动补零到6位
                    )
                    K_pred_depth = torch.from_numpy(
                        np.loadtxt(
                            os.path.join(scan["pred_depth_dir"], "intrinsic_depth.txt")
                        )[:3, :3]
                    ).float()
                else:
                    pred_depth_imgfile = None
                    K_pred_depth = None

                self.frames.append(
                    [
                        rgb_imgfiles[i],
                        pred_depth_imgfile,
                        poses[i],
                        K_color,
                        K_pred_depth,
                        scan["scan_name"],
                        keyframe,
                        initial_frame,
                        final_frame,
                        gt_origin,
                        gt_maxbound,
                    ]
                )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        (
            rgb_imgfile,
            pred_depth_imgfile,
            pose,
            K_color,
            K_pred_depth,
            scan_name,
            keyframe,
            initial_frame,
            final_frame,
            gt_origin,
            gt_maxbound,
        ) = self.frames[idx]

        rgb_img = load_rgb_imgs([rgb_imgfile], TARGET_RGB_IMG_SIZE)


        result = {
            "rgb_imgs": rgb_img,
            "poses": pose,
            "K_color": K_color,
            "scan_name": scan_name,
            "gt_origin": gt_origin,
            "gt_maxbound": gt_maxbound,
            "keyframe": keyframe,
            "initial_frame": initial_frame,
            "final_frame": final_frame,
        }
        if self.load_depth:
            # pred_depth_img = load_depth_img(pred_depth_imgfile, TARGET_RGB_IMG_SIZE)[None]
            #print(str(scan_name) + str(idx) +str(pred_depth_imgfile))
            pred_depth_img = load_depth_img(pred_depth_imgfile)[None]

            result["pred_depth_imgs"] = pred_depth_img
            result["K_pred_depth"] = K_pred_depth

        return result
    
    
    
class TartanAirDataset(torch.utils.data.Dataset):
    def __init__(self, scans, load_depth=True, gt_depth=False):
        self.scans = scans
        self.load_depth = load_depth
        self.gt_depth = gt_depth
        self.frames = []

        for scan in self.scans:
            try:
                # 1. 调用 load_scan_tartanair
                # rgb_files, depth_files (GT), poses, K_color, kf_idx
                rgb_files, gt_depth_files, poses, K_color, kf_idx = load_scan_tartanair(scan)
            except ValueError as e:
                print(f"警告：处理场景 {scan['scan_name']} 时出错。错误：{e}")
                continue

            # 2. 计算 Bound (保持原有的 margin 方法)
            margin = 4
            cam_centers = poses[:, :3, 3] 
            min_xyz = cam_centers.min(dim=0).values
            max_xyz = cam_centers.max(dim=0).values
            gt_origin = (min_xyz - margin)
            gt_maxbound = (max_xyz + margin)
            
            # 3. 准备每一帧的数据
            for i in range(len(rgb_files)):
                # --- 核心逻辑修改：深度路径分支 ---
                if self.load_depth:
                    if self.gt_depth:
                        # 情况 A: 使用 Ground Truth 深度 (.npy)
                        depth_file = gt_depth_files[i]
                        K_pred_depth = K_color # GT 深度通常与原图内参一致
                    else:
                        # 情况 B: 使用预测深度 (.png)
                        # 1. 获取带后缀的文件名，例如 "000165_left.png" -> "000165_left"
                        full_base_name = os.path.basename(rgb_files[i]).split('.')[0]

                        # 2. 进一步切割，去掉 "_left" 等后缀，只保留数字部分
                        # .split('_')[0] 会将 "000165_left" 切割为 ["000165", "left"] 并取第一个元素
                        pure_number = full_base_name.split('_')[0]

                        # 3. 自动补零到 6 位并拼接后缀 -> "000165.png"
                        depth_filename = pure_number.zfill(6) + ".png"

                        depth_file = os.path.join(
                            scan["pred_depth_dir"],
                            "depth",
                            depth_filename
                        )

                        # 建议：在调试阶段添加一个打印，确认第一帧路径是否正确
                        # if i == 0:
                        #     print(f"DEBUG: RGB文件={rgb_files[i]} -> 转换后的深度文件={depth_file}")
                        # 尝试加载预测深度的内参，如果不存在则退而求其次使用 K_color
                        intrinsic_path = os.path.join(scan["pred_depth_dir"], "intrinsic_depth.txt")
                        if os.path.exists(intrinsic_path):
                            K_pred_depth = torch.from_numpy(np.loadtxt(intrinsic_path)[:3, :3]).float()
                        else:
                            K_pred_depth = K_color
                else:
                    depth_file = None
                    K_pred_depth = None
                # --------------------------------

                self.frames.append({
                    "rgb_file": rgb_files[i],
                    "depth_file": depth_file,
                    "pose": poses[i],
                    "K": K_color,
                    "K_pred_depth": K_pred_depth,
                    "scan_name": scan["scan_name"],
                    "gt_origin": gt_origin,
                    "gt_maxbound": gt_maxbound,
                    "keyframe_index": kf_idx[i],
                    "is_initial": i == 0,
                    "is_final": i == len(rgb_files) - 1,
                })

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_data = self.frames[idx]
        
        # 加载 RGB
        rgb_img = load_rgb_imgs([frame_data["rgb_file"]], TARGET_RGB_IMG_SIZE)

        result = {
            "rgb_imgs": rgb_img,
            "poses": frame_data["pose"],
            "K_color": frame_data["K"],
            "scan_name": frame_data["scan_name"],
            "gt_origin": frame_data["gt_origin"],
            "gt_maxbound": frame_data["gt_maxbound"],
            "keyframe": True,
            "initial_frame": frame_data["is_initial"],
            "final_frame": frame_data["is_final"],
        }

        # --- 核心逻辑修改：深度加载分支 ---
        if self.load_depth:
            if self.gt_depth:
                # 加载 GT 深度 (.npy)
                depth = np.load(frame_data["depth_file"]).astype(np.float32)
                result["pred_depth_imgs"] = torch.from_numpy(depth).unsqueeze(0) 
            else:
                # 加载预测深度 (.png)，参考 TNTDataset
                # 使用你参考代码中的 load_depth_img 函数
                pred_depth_img = load_depth_img(frame_data["depth_file"])[None]
                result["pred_depth_imgs"] = pred_depth_img

            result["K_pred_depth"] = frame_data["K_pred_depth"]

        return result


#for moge2    
class Moge2Dataset(torch.utils.data.Dataset):
    def __init__(self, scans):
        self.scans = scans
        self.frames = []

        # 遍历所有场景并预加载元数据
        for scan in self.scans:
            try:
                rgb_files, depth_files, poses, scales, biases, K ,estimate_scale= load_scan_moge2(scan)
            except Exception as e:
                print(f"加载场景 {scan['scan_name']} 失败: {e}")
                continue

            # 计算场景边界
            mx=0
            my=2
            mz=0
            margin = torch.tensor([mx, my, mz], dtype=torch.float32)
            cam_centers = torch.from_numpy(poses[:, :3, 3]).float()
            min_xyz = cam_centers.min(dim=0).values
            max_xyz = cam_centers.max(dim=0).values
            gt_origin = (min_xyz - margin)
            gt_maxbound = (max_xyz + margin)

            # 将每一帧的信息打包存入 self.frames
            for i in range(len(rgb_files)):
                self.frames.append({
                    "rgb_file": rgb_files[i],
                    "depth_file": depth_files[i],
                    "pose": torch.from_numpy(poses[i]).float(),
                    "scale": float(scales[i]),
                    "bias": float(biases[i]),
                    "K": K,
                    "scan_name": scan["scan_name"],
                    "gt_origin": gt_origin,
                    "gt_maxbound": gt_maxbound,
                    "is_initial": i == 0,
                    "is_final": i == len(rgb_files) - 1,
                    "estimate_scale":estimate_scale
                })

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_data = self.frames[idx]
        
        # 1. 加载 RGB
        rgb_img = load_rgb_imgs([frame_data["rgb_file"]], TARGET_RGB_IMG_SIZE)

        # 2. 加载并处理深度图 (应用 Scale 和 Bias)
        # 这里动态读取，节省内存
        pred_depth_img = load_depth_with_scale_and_bias(
            frame_data["depth_file"], 
            frame_data["scale"], 
            frame_data["bias"],
            frame_data["estimate_scale"]
        )

        result = {
            "rgb_imgs": rgb_img,                # Tensor (1, 3, H, W) 或 (3, H, W) 取决于你的 load_rgb_imgs 实现
            "pred_depth_imgs": pred_depth_img,  # Tensor (1, H, W) 已经是真实尺度
            "poses": frame_data["pose"],        # Tensor (4, 4)
            "K_color": frame_data["K"],         # Tensor (3, 3)
            "K_pred_depth": frame_data["K"],    # 假设深度图内参同RGB
            
            # 元数据
            "scan_name": frame_data["scan_name"],
            "gt_origin": frame_data["gt_origin"],
            "gt_maxbound": frame_data["gt_maxbound"],
            "initial_frame": frame_data["is_initial"],
            "final_frame": frame_data["is_final"],
            
            # 如果你需要 keyframe 逻辑，可以在这里硬编码 True 或者在 init 里计算
            "keyframe": True 
        }

        return result
    
    
class Moge2Dataset_NoOptimize(torch.utils.data.Dataset):
    def __init__(self, scans):
        self.scans = scans
        self.frames = []

        # 遍历所有场景并预加载元数据
        for scan in self.scans:
            try:
                rgb_files, depth_files, poses,  K, estimate_scale = load_scan_moge2_no_optimize(scan)
            except Exception as e:
                print(f"加载场景 {scan['scan_name']} 失败: {e}")
                continue

            # 计算场景边界
            mx=0
            my=2
            mz=0
            margin = torch.tensor([mx, my, mz], dtype=torch.float32)
            cam_centers = torch.from_numpy(poses[:, :3, 3]).float()
            min_xyz = cam_centers.min(dim=0).values
            max_xyz = cam_centers.max(dim=0).values
            gt_origin = (min_xyz - margin)
            gt_maxbound = (max_xyz + margin)

            # 将每一帧的信息打包存入 self.frames
            for i in range(len(rgb_files)):
                self.frames.append({
                    "rgb_file": rgb_files[i],
                    "depth_file": depth_files[i],
                    "pose": torch.from_numpy(poses[i]).float(),
                    "K": K,
                    "scan_name": scan["scan_name"],
                    "gt_origin": gt_origin,
                    "gt_maxbound": gt_maxbound,
                    "is_initial": i == 0,
                    "is_final": i == len(rgb_files) - 1,
                    "estimate_scale":estimate_scale
                })

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_data = self.frames[idx]
        
        # 1. 加载 RGB
        rgb_img = load_rgb_imgs([frame_data["rgb_file"]], TARGET_RGB_IMG_SIZE)

        # 2. 加载并处理深度图 (应用 Scale 和 Bias)
        # 这里动态读取，节省内存
        
        # depth = np.load(frame_data["depth_file"]).astype(np.float32)
        depth_raw = cv2.imread(frame_data["depth_file"], -1)
        if depth_raw is None:
            raise FileNotFoundError(f"无法读取深度图")

        # 2. 转为 float32
        depth_val = depth_raw.astype(np.float32)
        
        depth_val/=1000
        depth_val*=frame_data["estimate_scale"]
        
        pred_depth_img = torch.from_numpy(depth_val).unsqueeze(0) 

        result = {
            "rgb_imgs": rgb_img,                # Tensor (1, 3, H, W) 或 (3, H, W) 取决于你的 load_rgb_imgs 实现
            "pred_depth_imgs": pred_depth_img,  # Tensor (1, H, W) 已经是真实尺度
            "poses": frame_data["pose"],        # Tensor (4, 4)
            "K_color": frame_data["K"],         # Tensor (3, 3)
            "K_pred_depth": frame_data["K"],    # 假设深度图内参同RGB
            
            # 元数据
            "scan_name": frame_data["scan_name"],
            "gt_origin": frame_data["gt_origin"],
            "gt_maxbound": frame_data["gt_maxbound"],
            "initial_frame": frame_data["is_initial"],
            "final_frame": frame_data["is_final"],
            
            # 如果你需要 keyframe 逻辑，可以在这里硬编码 True 或者在 init 里计算
            "keyframe": True 
        }

        return result

