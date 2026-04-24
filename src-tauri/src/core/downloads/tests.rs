use super::helpers::*;
use super::models::*;
use crate::core::filesystem::helpers::resolve_path_within_jan_data_folder;
use reqwest::header::HeaderMap;
use std::collections::HashMap;

// Helper function to create a minimal proxy config for testing
fn create_test_proxy_config(url: &str) -> ProxyConfig {
    ProxyConfig {
        url: url.to_string(),
        username: None,
        password: None,
        no_proxy: None,
        ignore_ssl: None,
    }
}

#[test]
fn test_convert_headers() {
    let mut headers = HashMap::new();
    headers.insert("User-Agent".to_string(), "test-agent".to_string());
    headers.insert("Authorization".to_string(), "Bearer token".to_string());

    let header_map = _convert_headers(&headers).unwrap();
    assert_eq!(header_map.len(), 2);
    assert_eq!(header_map.get("User-Agent").unwrap(), "test-agent");
    assert_eq!(header_map.get("Authorization").unwrap(), "Bearer token");
}

#[test]
fn test_download_item_with_ssl_proxy() {
    // Test that DownloadItem can be created with SSL proxy configuration
    let mut proxy_config = create_test_proxy_config("https://proxy.example.com:8080");
    proxy_config.ignore_ssl = Some(true);

    let download_item = DownloadItem {
        url: "https://example.com/file.zip".to_string(),
        save_path: "downloads/file.zip".to_string(),
        proxy: Some(proxy_config),
        sha256: None,
        size: None,
        model_id: None,
    };

    assert!(download_item.proxy.is_some());
    let proxy = download_item.proxy.unwrap();
    assert_eq!(proxy.ignore_ssl, Some(true));
}

#[test]
fn test_client_creation_with_ssl_settings() {
    // Test client creation with SSL settings
    let mut proxy_config = create_test_proxy_config("https://proxy.example.com:8080");
    proxy_config.ignore_ssl = Some(true);

    let download_item = DownloadItem {
        url: "https://example.com/file.zip".to_string(),
        save_path: "downloads/file.zip".to_string(),
        proxy: Some(proxy_config),
        sha256: None,
        size: None,
        model_id: None,
    };

    let header_map = HeaderMap::new();
    let result = _get_client_for_item(&download_item, &header_map);

    // Should create client successfully even with SSL settings
    assert!(result.is_ok());
}

#[test]
fn test_download_item_creation() {
    let item = DownloadItem {
        url: "https://example.com/file.tar.gz".to_string(),
        save_path: "models/test.tar.gz".to_string(),
        proxy: None,
        sha256: None,
        size: None,
        model_id: None,
    };

    assert_eq!(item.url, "https://example.com/file.tar.gz");
    assert_eq!(item.save_path, "models/test.tar.gz");
}

#[cfg(unix)]
#[test]
fn test_download_scope_accepts_absolute_path_inside_canonical_root() {
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base_dir = std::env::temp_dir().join(format!("jan-download-scope-{unique}"));
    let configured_root = base_dir.join("home").join("user").join("jan-data");
    let canonical_root = base_dir
        .join("var")
        .join("home")
        .join("user")
        .join("jan-data");
    fs::create_dir_all(&canonical_root).unwrap();
    fs::create_dir_all(configured_root.parent().unwrap()).unwrap();
    symlink(&canonical_root, &configured_root).unwrap();

    let candidate = canonical_root.join("llamacpp/backends/v1/backend.tar.gz");
    let (_, resolved_path) =
        resolve_path_within_jan_data_folder(&configured_root, candidate.to_string_lossy().as_ref())
            .unwrap();

    let expected_path = canonical_root
        .canonicalize()
        .unwrap()
        .join("llamacpp/backends/v1/backend.tar.gz");
    assert_eq!(resolved_path, expected_path);

    let _ = fs::remove_dir_all(&base_dir);
}

#[test]
fn test_download_event_creation() {
    let event = DownloadEvent {
        transferred: 1024,
        total: 2048,
    };

    assert_eq!(event.transferred, 1024);
    assert_eq!(event.total, 2048);
}

#[test]
fn test_err_to_string() {
    let error = "Test error";
    let result = err_to_string(error);
    assert_eq!(result, "Error: Test error");
}

#[test]
fn test_convert_headers_valid() {
    let mut headers = HashMap::new();
    headers.insert("Content-Type".to_string(), "application/json".to_string());
    headers.insert("Authorization".to_string(), "Bearer token123".to_string());

    let result = _convert_headers(&headers);
    assert!(result.is_ok());

    let header_map = result.unwrap();
    assert_eq!(header_map.len(), 2);
    assert_eq!(header_map.get("Content-Type").unwrap(), "application/json");
    assert_eq!(header_map.get("Authorization").unwrap(), "Bearer token123");
}

#[test]
fn test_convert_headers_invalid_header_name() {
    let mut headers = HashMap::new();
    headers.insert("Invalid\nHeader".to_string(), "value".to_string());

    let result = _convert_headers(&headers);
    assert!(result.is_err());
}

#[test]
fn test_convert_headers_invalid_header_value() {
    let mut headers = HashMap::new();
    headers.insert("Content-Type".to_string(), "invalid\nvalue".to_string());

    let result = _convert_headers(&headers);
    assert!(result.is_err());
}

#[test]
fn test_download_manager_state_default() {
    let state = DownloadManagerState::default();
    assert!(state.cancel_tokens.is_empty());
}

#[test]
fn test_download_event_serialization() {
    let event = DownloadEvent {
        transferred: 512,
        total: 1024,
    };

    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"transferred\":512"));
    assert!(json.contains("\"total\":1024"));
}

#[test]
fn test_download_item_deserialization() {
    let json = r#"{"url":"https://example.com/file.zip","save_path":"downloads/file.zip"}"#;
    let item: DownloadItem = serde_json::from_str(json).unwrap();

    assert_eq!(item.url, "https://example.com/file.zip");
    assert_eq!(item.save_path, "downloads/file.zip");
}

// ===== convert_to_mirror_url =====

#[test]
fn test_convert_to_mirror_url_huggingface() {
    let url = "https://huggingface.co/some/repo/resolve/main/model.gguf";
    let mirror = convert_to_mirror_url(url).expect("should produce a mirror url");
    assert!(mirror.starts_with("https://apps") && mirror.contains(".jan.ai/"));
    assert!(mirror.ends_with("huggingface.co/some/repo/resolve/main/model.gguf"));
}

#[test]
fn test_convert_to_mirror_url_huggingface_subdomain() {
    // Subdomains of mirror domains should also be mirrored
    let url = "https://cdn.huggingface.co/file.bin";
    let mirror = convert_to_mirror_url(url).expect("subdomain should mirror");
    assert!(mirror.ends_with("cdn.huggingface.co/file.bin"));
}

#[test]
fn test_convert_to_mirror_url_http_scheme() {
    let url = "http://huggingface.co/file";
    let mirror = convert_to_mirror_url(url).expect("http should be stripped too");
    assert!(mirror.ends_with("huggingface.co/file"));
    assert!(!mirror.contains("http://huggingface.co"));
}

#[test]
fn test_convert_to_mirror_url_non_mirror_domain() {
    assert!(convert_to_mirror_url("https://example.com/file.bin").is_none());
    assert!(convert_to_mirror_url("https://github.com/x/y").is_none());
}

#[test]
fn test_convert_to_mirror_url_invalid_url() {
    assert!(convert_to_mirror_url("not a url").is_none());
    assert!(convert_to_mirror_url("").is_none());
}

#[test]
fn test_convert_to_mirror_url_not_substring_match() {
    // A domain that merely contains "huggingface.co" as substring (not as suffix) must NOT match
    let url = "https://huggingface.co.evil.com/file";
    assert!(convert_to_mirror_url(url).is_none());
}

// ===== err_to_string =====

#[test]
fn test_err_to_string_with_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
    let s = err_to_string(io_err);
    assert!(s.starts_with("Error: "));
    assert!(s.contains("missing"));
}

// ===== _convert_headers edge cases =====

#[test]
fn test_convert_headers_empty() {
    let headers: HashMap<String, String> = HashMap::new();
    let result = _convert_headers(&headers).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_convert_headers_empty_value_is_ok() {
    // Empty header values are valid per HTTP spec
    let mut headers = HashMap::new();
    headers.insert("X-Empty".to_string(), "".to_string());
    let result = _convert_headers(&headers).unwrap();
    assert_eq!(result.get("X-Empty").unwrap(), "");
}

#[test]
fn test_convert_headers_invalid_name_with_space() {
    let mut headers = HashMap::new();
    headers.insert("Bad Header".to_string(), "v".to_string());
    assert!(_convert_headers(&headers).is_err());
}

// ===== _get_client_for_item =====

#[test]
fn test_get_client_for_item_no_proxy() {
    let item = DownloadItem {
        url: "https://example.com/file".to_string(),
        save_path: "x".to_string(),
        proxy: None,
        sha256: None,
        size: None,
        model_id: None,
    };
    assert!(_get_client_for_item(&item, &HeaderMap::new()).is_ok());
}

#[test]
fn test_get_client_for_item_with_bypassed_proxy() {
    // Proxy configured but URL is in no_proxy list → still builds client without contacting proxy
    let mut proxy = create_test_proxy_config("http://proxy.example.com:8080");
    proxy.no_proxy = Some(vec!["example.com".to_string()]);
    let item = DownloadItem {
        url: "https://example.com/file".to_string(),
        save_path: "x".to_string(),
        proxy: Some(proxy),
        sha256: None,
        size: None,
        model_id: None,
    };
    assert!(_get_client_for_item(&item, &HeaderMap::new()).is_ok());
}

#[test]
fn test_get_client_for_item_invalid_proxy_url() {
    let item = DownloadItem {
        url: "https://example.com/file".to_string(),
        save_path: "x".to_string(),
        proxy: Some(create_test_proxy_config("not-a-url")),
        sha256: None,
        size: None,
        model_id: None,
    };
    assert!(_get_client_for_item(&item, &HeaderMap::new()).is_err());
}

// ===== ProgressTracker =====

#[tokio::test]
async fn test_progress_tracker_initial_total() {
    let mut sizes = HashMap::new();
    sizes.insert("a".to_string(), 100u64);
    sizes.insert("b".to_string(), 250u64);
    let tracker = ProgressTracker::new(&[], sizes);
    let (transferred, total) = tracker.get_total_progress().await;
    assert_eq!(transferred, 0);
    assert_eq!(total, 350);
}

#[tokio::test]
async fn test_progress_tracker_update_and_sum() {
    let mut sizes = HashMap::new();
    sizes.insert("u1".to_string(), 1000u64);
    let tracker = ProgressTracker::new(&[], sizes);

    tracker.update_progress("file-0", 200).await;
    tracker.update_progress("file-1", 300).await;
    let (transferred, total) = tracker.get_total_progress().await;
    assert_eq!(transferred, 500);
    assert_eq!(total, 1000);

    // Update overwrites previous value for the same file id
    tracker.update_progress("file-0", 400).await;
    let (transferred, _) = tracker.get_total_progress().await;
    assert_eq!(transferred, 700);
}

#[tokio::test]
async fn test_progress_tracker_add_to_total() {
    let tracker = ProgressTracker::new(&[], HashMap::new());
    let (_, total0) = tracker.get_total_progress().await;
    assert_eq!(total0, 0);

    tracker.add_to_total(1024);
    tracker.add_to_total(2048);
    let (_, total) = tracker.get_total_progress().await;
    assert_eq!(total, 3072);
}

#[tokio::test]
async fn test_progress_tracker_clone_shares_state() {
    let mut sizes = HashMap::new();
    sizes.insert("x".to_string(), 500u64);
    let tracker = ProgressTracker::new(&[], sizes);
    let clone = tracker.clone();

    clone.update_progress("f", 123).await;
    clone.add_to_total(100);

    let (transferred, total) = tracker.get_total_progress().await;
    assert_eq!(transferred, 123);
    assert_eq!(total, 600);
}

// ===== DownloadEvent / DownloadItem serde =====

#[test]
fn test_download_item_full_deserialization() {
    let json = r#"{
        "url": "https://huggingface.co/m/file.gguf",
        "save_path": "models/m/file.gguf",
        "sha256": "abc123",
        "size": 4096,
        "model_id": "m"
    }"#;
    let item: DownloadItem = serde_json::from_str(json).unwrap();
    assert_eq!(item.sha256.as_deref(), Some("abc123"));
    assert_eq!(item.size, Some(4096));
    assert_eq!(item.model_id.as_deref(), Some("m"));
    assert!(item.proxy.is_none());
}

#[test]
fn test_download_event_zero_values() {
    let evt = DownloadEvent {
        transferred: 0,
        total: 0,
    };
    let json = serde_json::to_string(&evt).unwrap();
    assert_eq!(json, r#"{"transferred":0,"total":0}"#);
}

#[test]
fn test_download_manager_state_default_is_empty() {
    let s = DownloadManagerState::default();
    assert_eq!(s.cancel_tokens.len(), 0);
}
