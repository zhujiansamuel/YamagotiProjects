# batch_generate_psta_same_ts 详细流程图

> 说明：下图基于 `AppleStockChecker.tasks.timestamp_alignment_task.batch_generate_psta_same_ts` 的实际实现，覆盖参数初始化、数据收集、聚合策略判断、子任务分发、顺序/并发执行与结果汇总的完整路径。

```mermaid
flowchart TD
    start([batch_generate_psta_same_ts 入口])

    start --> init["初始化任务参数\n- task_job_id = job_id or request.id\n- ts_iso = timestamp_iso or nearest_past_minute_iso()\n- MODE = agg_mode.lower() or boundary"]

    init --> collect["collect_items_for_psta\n- window_minutes\n- timestamp_iso\n- shop_ids / iphone_ids\n- max_items"]

    collect --> pack{pack 是否为空?}
    pack -->|空| pack_default["pack = [{}]\nrows = []\nbucket_minute_key = {}"]
    pack -->|有| pack_extract["rows = pack['rows']\nbucket_minute_key = pack['bucket_minute_key']"]

    pack_default --> ctx
    pack_extract --> ctx

    ctx["计算聚合上下文 ctx\n- dt0 = _to_aware(ts_iso)\n- step0 = _floor_to_step(dt0, agg_minutes)\n- bucket_start / bucket_end\n- agg_mode / force_agg"]

    ctx --> notify_ctx["notify_progress_all(agg_ctx)"]

    notify_ctx --> loop_start{遍历 bucket_minute_key?}

    loop_start -->|无| no_subtasks["subtasks = []"]

    loop_start -->|有| loop_build["按 minute_iso 遍历\n构建 minute_rows"]

    loop_build --> build_rows["minute_rows 由 rows 索引组装\n字段: shop_id/iphone_id/recorded_at/price_new"]

    build_rows --> calc_boundary["mdt = _to_aware(minute_iso)\nboundary = _floor_to_step(mdt, agg_minutes)\nis_boundary = (mdt == boundary)"]

    calc_boundary --> mode_switch{MODE 判断}

    mode_switch -->|off| mode_off["do_agg_local = False\nagg_start_iso = None"]
    mode_switch -->|rolling| mode_rolling["do_agg_local = True\nagg_start_iso = _rolling_start(mdt, agg_minutes)"]
    mode_switch -->|boundary| mode_boundary["do_agg_local = is_boundary\nagg_start_iso = boundary"]

    mode_off --> append_check
    mode_rolling --> append_check
    mode_boundary --> append_check

    append_check{minute_rows 有数据\n或 do_agg_local=True?}
    append_check -->|否| skip_bucket["跳过该分钟"]
    append_check -->|是| append_task["追加子任务\n psta_process_minute_bucket.s\n- ts_iso=minute_iso\n- rows=minute_rows\n- do_agg/do_agg_local\n- agg_start_iso/agg_minutes\n- job_id/task_ver"]

    append_task --> loop_build
    skip_bucket --> loop_build

    no_subtasks --> notify_dispatch
    loop_build --> notify_dispatch

    notify_dispatch["notify_progress_all(\n status=running,\n step=dispatch_buckets,\n buckets=len(subtasks))"]

    notify_dispatch --> empty_check{len(subtasks) == 0?}

    empty_check -->|是| return_empty["返回空结果\n并 notify_progress_all(done)\nsummary: total_buckets=0"]

    empty_check -->|否| exec_mode{sequential?}

    exec_mode -->|是| sequential_exec["顺序执行:\nfor subtask in subtasks\n- subtask.apply().get()\n- notify_progress_all(进度)\n- 错误记录不中断"]

    sequential_exec --> finalize_seq["psta_finalize_buckets(results,\n job_id, ts_iso, agg_ctx)"]
    finalize_seq --> return_seq["返回 {timestamp, total_buckets,\n job_id, sequential=True, result}"]

    exec_mode -->|否| parallel_exec["并发执行:\ncallback = psta_finalize_buckets.s\nchord(subtasks)(callback)"]

    parallel_exec --> return_parallel["返回 {timestamp, total_buckets,\n job_id, chord_id}"]

    return_empty --> end([结束])
    return_seq --> end
    return_parallel --> end

    classDef decision fill:#fff3bf,stroke:#f08c00,color:#000;
    class pack,loop_start,mode_switch,append_check,empty_check,exec_mode decision;
```

## 关键说明
- **聚合模式 (agg_mode)**
  - `off`：不做聚合，仅处理分钟行。
  - `rolling`：滚动窗口聚合，每分钟都聚合，起点为 `_rolling_start`。
  - `boundary`：仅边界分钟聚合（默认）。
- **空分钟策略**：若 `minute_rows` 为空但 `do_agg_local=True`（例如边界分钟），仍会下发“仅聚合”的子任务。
- **执行方式**
  - `sequential=True`：逐个同步执行子任务，实时上报进度，错误记录但不中断。
  - `sequential=False`：默认并发模式，使用 Celery chord 汇总。
