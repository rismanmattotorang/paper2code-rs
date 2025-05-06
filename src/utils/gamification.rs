// src/utils/gamification.rs
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;
use std::time::{Duration, Instant};
use chrono::{DateTime, Local};

use crate::error::AppError;
use crate::llm::strategy::TaskType;

/// Custom type for timestamps that can be serialized
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timestamp {
    #[serde(with = "timestamp_serde")]
    pub time: Instant,
}

mod timestamp_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let system_now = SystemTime::now();
        let instant_now = Instant::now();
        let duration = instant_now.duration_since(*instant);
        let system_instant = system_now - duration;
        let timestamp = system_instant
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_secs();
        serializer.serialize_u64(timestamp)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        let timestamp = u64::deserialize(deserializer)?;
        let system_time = UNIX_EPOCH + Duration::from_secs(timestamp);
        let now = SystemTime::now();
        let duration = now.duration_since(system_time).unwrap_or(Duration::from_secs(0));
        Ok(Instant::now() - duration)
    }
}

/// Gamification system for code generation
#[derive(Debug, Clone)]
pub struct GamificationSystem {
    /// User's gamification profile
    profile: Arc<RwLock<UserProfile>>,
    
    /// Path to save profile data
    save_path: Option<String>,
    
    /// Whether gamification is enabled
    enabled: bool,
}

/// User's gamification profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// Unique user ID
    pub id: String,
    
    /// User's current level
    pub level: u32,
    
    /// Experience points towards next level
    pub xp: u64,
    
    /// XP needed for next level
    pub next_level_xp: u64,
    
    /// Completed achievements
    pub achievements: Vec<Achievement>,
    
    /// Statistics for different tasks
    pub stats: HashMap<String, u64>,
    
    /// LLM performance metrics
    pub llm_metrics: HashMap<String, f64>,
    
    /// Streak information
    pub streak: StreakInfo,
    
    /// Progress on current challenges
    pub challenges: Vec<Challenge>,
    
    /// Generated code quality metrics
    pub code_quality: HashMap<String, f64>,
}

/// Streak information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreakInfo {
    /// Number of consecutive days using the tool
    pub days: u32,
    
    /// Last date used (ISO format)
    pub last_date: String,
    
    /// Longest streak achieved
    pub longest: u32,
}

/// Achievement types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AchievementType {
    FirstCodeGeneration,
    FirstSuccessfulGeneration,
    FirstErrorRecovery,
    FirstParallelProcessing,
    FirstOptimization,
    FirstGamification,
    FirstCustomization,
    FirstIntegration,
    FirstDeployment,
    FirstMonitoring,
}

/// Achievement status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    pub achievement_type: AchievementType,
    pub unlocked_at: Option<DateTime<Local>>,
    pub progress: f64,
    pub description: String,
}

impl Achievement {
    pub fn new(achievement_type: AchievementType, description: String) -> Self {
        Self {
            achievement_type,
            unlocked_at: None,
            progress: 0.0,
            description,
        }
    }
    
    pub fn is_unlocked(&self) -> bool {
        self.unlocked_at.is_some()
    }
    
    pub fn update_progress(&mut self, progress: f64) {
        self.progress = progress.min(1.0);
        if self.progress >= 1.0 && self.unlocked_at.is_none() {
            self.unlocked_at = Some(Local::now());
        }
    }
}

/// User progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProgress {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub total_duration: Duration,
    pub achievements: HashMap<AchievementType, Achievement>,
    pub level: u32,
    pub experience: u64,
}

impl UserProgress {
    pub fn new() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            total_duration: Duration::from_secs(0),
            achievements: HashMap::new(),
            level: 1,
            experience: 0,
        }
    }
    
    pub fn record_operation(&mut self, success: bool, duration: Duration) {
        self.total_operations += 1;
        if success {
            self.successful_operations += 1;
        } else {
            self.failed_operations += 1;
        }
        self.total_duration += duration;
        
        // Award experience points
        let base_xp = if success { 10 } else { 5 };
        let duration_bonus = (duration.as_secs_f64() * 2.0) as u64;
        self.experience += base_xp + duration_bonus;
        
        // Check for level up
        let xp_for_next_level = self.level * 100;
        if self.experience >= xp_for_next_level.into() {
            self.level += 1;
        }
    }
    
    pub fn unlock_achievement(&mut self, achievement_type: AchievementType) {
        if let Some(achievement) = self.achievements.get_mut(&achievement_type) {
            achievement.update_progress(1.0);
        }
    }
    
    pub fn update_achievement_progress(&mut self, achievement_type: AchievementType, progress_value: f64) {
        if let Some(achievement) = self.achievements.get_mut(&achievement_type) {
            achievement.update_progress(progress_value);
        }
    }
}

/// Gamification manager
pub struct GamificationManager {
    progress: Arc<RwLock<UserProgress>>,
}

impl GamificationManager {
    pub fn new() -> Self {
        Self {
            progress: Arc::new(RwLock::new(UserProgress::new())),
        }
    }
    
    pub async fn record_operation(&self, success: bool, duration: Duration) {
        let mut progress = self.progress.write().await;
        progress.record_operation(success, duration);
    }
    
    pub async fn unlock_achievement(&self, achievement_type: AchievementType) {
        let mut progress = self.progress.write().await;
        progress.unlock_achievement(achievement_type);
    }
    
    pub async fn update_achievement_progress(&self, achievement_type: AchievementType, progress_value: f64) {
        let mut progress = self.progress.write().await;
        progress.update_achievement_progress(achievement_type, progress_value);
    }
    
    pub async fn get_progress(&self) -> UserProgress {
        self.progress.read().await.clone()
    }
    
    pub async fn get_achievements(&self) -> HashMap<AchievementType, Achievement> {
        self.progress.read().await.achievements.clone()
    }
    
    pub async fn get_level(&self) -> u32 {
        self.progress.read().await.level
    }
    
    pub async fn get_experience(&self) -> u64 {
        self.progress.read().await.experience
    }
}

/// Achievement tracker
pub struct AchievementTracker {
    manager: Arc<GamificationManager>,
}

impl AchievementTracker {
    pub fn new(manager: Arc<GamificationManager>) -> Self {
        Self { manager }
    }
    
    pub async fn track_operation<F, Fut, T>(
        &self,
        operation: F,
    ) -> std::result::Result<T, AppError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, AppError>>,
    {
        let start_time = Instant::now();
        let result = operation().await;
        let duration = start_time.elapsed();
        
        self.manager.record_operation(result.is_ok(), duration).await;
        
        result
    }
    
    pub async fn track_achievement<F, Fut, T>(
        &self,
        achievement_type: AchievementType,
        operation: F,
    ) -> std::result::Result<T, AppError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, AppError>>,
    {
        let result = self.track_operation(operation).await;
        
        if result.is_ok() {
            self.manager.unlock_achievement(achievement_type).await;
        }
        
        result
    }
}

/// Current challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenge {
    /// Challenge ID
    pub id: String,
    
    /// Challenge name
    pub name: String,
    
    /// Challenge description
    pub description: String,
    
    /// Current progress
    pub progress: u32,
    
    /// Target to complete
    pub target: u32,
    
    /// XP reward when completed
    pub xp_reward: u64,
    
    /// Whether the challenge is completed
    pub completed: bool,
}

impl Default for UserProfile {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            level: 1,
            xp: 0,
            next_level_xp: 100,
            achievements: Vec::new(),
            stats: HashMap::new(),
            llm_metrics: HashMap::new(),
            streak: StreakInfo {
                days: 0,
                last_date: chrono::Local::now().date_naive().to_string(),
                longest: 0,
            },
            challenges: vec![
                Challenge {
                    id: "first_code".to_string(),
                    name: "Code Creator".to_string(),
                    description: "Generate your first code from a research paper".to_string(),
                    progress: 0,
                    target: 1,
                    xp_reward: 50,
                    completed: false,
                },
                Challenge {
                    id: "process_10_pdfs".to_string(),
                    name: "PDF Master".to_string(),
                    description: "Process 10 PDF research papers".to_string(),
                    progress: 0,
                    target: 10,
                    xp_reward: 200,
                    completed: false,
                },
                Challenge {
                    id: "use_both_llms".to_string(),
                    name: "LLM Explorer".to_string(),
                    description: "Try all different LLM strategies".to_string(),
                    progress: 0,
                    target: 5,
                    xp_reward: 150,
                    completed: false,
                },
            ],
            code_quality: HashMap::new(),
        }
    }
}

impl GamificationSystem {
    /// Create a new gamification system
    pub fn new(save_path: Option<String>, enabled: bool) -> Self {
        Self {
            profile: Arc::new(RwLock::new(UserProfile::default())),
            save_path,
            enabled,
        }
    }
    
    /// Load user profile from file
    pub async fn load_profile(&self) -> Result<(), AppError> {
        if !self.enabled {
            return Ok(());
        }
        
        if let Some(path) = &self.save_path {
            let path = Path::new(path);
            if path.exists() {
                let data = fs::read_to_string(path).await?;
                match serde_json::from_str::<UserProfile>(&data) {
                    Ok(profile) => {
                        let mut user_profile = self.profile.write().await;
                        *user_profile = profile;
                        
                        // Update streak if needed
                        self.update_streak().await?;
                        
                        info!("Loaded gamification profile from {}", path.display());
                        Ok(())
                    },
                    Err(e) => {
                        warn!("Failed to parse gamification profile: {}", e);
                        
                        // If corrupted, start fresh
                        let mut user_profile = self.profile.write().await;
                        *user_profile = UserProfile::default();
                        
                        Ok(())
                    }
                }
            } else {
                // No existing profile, create a new one
                let user_profile = UserProfile::default();
                let mut profile_lock = self.profile.write().await;
                *profile_lock = user_profile;
                
                // Save the new profile
                self.save_profile().await?;
                
                info!("Created new gamification profile");
                Ok(())
            }
        } else {
            // No save path, just use default profile
            Ok(())
        }
    }
    
    /// Save user profile to file
    pub async fn save_profile(&self) -> Result<(), AppError> {
        if !self.enabled {
            return Ok(());
        }
        
        if let Some(path) = &self.save_path {
            let profile = self.profile.read().await;
            let data = serde_json::to_string_pretty(&*profile)?;
            
            // Ensure directory exists
            if let Some(parent) = Path::new(path).parent() {
                if !parent.exists() {
                    fs::create_dir_all(parent).await?;
                }
            }
            
            fs::write(path, data).await?;
            debug!("Saved gamification profile to {}", path);
            Ok(())
        } else {
            // No save path, can't save
            Ok(())
        }
    }
    
    /// Award XP to the user
    pub async fn award_xp(&self, amount: u64, reason: &str) -> Result<bool, AppError> {
        if !self.enabled {
            return Ok(false);
        }
        
        let mut profile = self.profile.write().await;
        profile.xp += amount;
        
        // Check for level up
        let leveled_up = profile.xp >= profile.next_level_xp;
        if leveled_up {
            profile.level += 1;
            profile.xp -= profile.next_level_xp;
            
            // Increase XP requirement for next level (20% increase per level)
            profile.next_level_xp = (profile.next_level_xp as f64 * 1.2).ceil() as u64;
            
            info!("Level up! Now at level {}", profile.level);
        }
        
        debug!("Awarded {} XP for {}", amount, reason);
        drop(profile);
        
        self.save_profile().await?;
        
        Ok(leveled_up)
    }
    
    /// Update user stats
    pub async fn update_stat(&self, stat_name: &str, value: u64) -> Result<(), AppError> {
        if !self.enabled {
            return Ok(());
        }
        
        // Update the stat value
        {
            let mut profile = self.profile.write().await;
            let current = profile.stats.entry(stat_name.to_string()).or_insert(0);
            *current += value;
        }
        
        // Collect strategy_keys once if needed
        let strategy_keys = if stat_name.starts_with("llm_strategy_") {
            let profile = self.profile.read().await;
            profile.stats.keys()
                .filter(|k| k.starts_with("llm_strategy_"))
                .cloned()
                .collect::<Vec<String>>()
        } else {
            Vec::new()
        };
        
        // Get current stat value for reference
        let current_value = {
            let profile = self.profile.read().await;
            *profile.stats.get(stat_name).unwrap_or(&0)
        };
        
        // Collect challenge status changes
        let mut completed_challenges = Vec::new();
        let mut new_achievements = Vec::new();
        
        {
            let mut profile = self.profile.write().await;
            
            for challenge in &mut profile.challenges {
                if !challenge.completed {
                    match challenge.id.as_str() {
                        "first_code" if stat_name == "code_generated" => {
                            challenge.progress = challenge.progress.max(1);
                        },
                        "process_10_pdfs" if stat_name == "pdfs_processed" => {
                            challenge.progress = (current_value).min(challenge.target as u64) as u32;
                        },
                        "use_both_llms" if stat_name.starts_with("llm_strategy_") => {
                            challenge.progress = (strategy_keys.len() as u32).min(challenge.target);
                        },
                        _ => {}
                    }
                    
                    // Check if challenge is now completed
                    if challenge.progress >= challenge.target && !challenge.completed {
                        challenge.completed = true;
                        completed_challenges.push((challenge.name.clone(), challenge.xp_reward));
                        
                        // Store achievement info for later
                        new_achievements.push(Achievement {
                            achievement_type: AchievementType::FirstCodeGeneration,
                            unlocked_at: None,
                            progress: 1.0,
                            description: challenge.description.clone(),
                        });
                    }
                }
            }
        }
        
        // Award XP for completed challenges
        for (name, xp) in completed_challenges {
            self.award_xp(xp, &format!("Completed challenge: {}", name)).await?;
        }
        
        // Check for achievements based on stats
        self.check_stat_achievements(stat_name).await?;
        
        self.save_profile().await?;
        
        Ok(())
    }
    
    /// Track LLM usage and performance
    pub async fn track_llm_usage(
        &self,
        llm_name: &str,
        task_type: TaskType,
        success: bool,
        duration_ms: u64,
    ) -> Result<(), AppError> {
        if !self.enabled {
            return Ok(());
        }
        
        // Update stats
        self.update_stat(&format!("llm_usage_{}", llm_name), 1).await?;
        
        if success {
            self.update_stat(&format!("llm_success_{}", llm_name), 1).await?;
        }
        
        self.update_stat(&format!("task_{:?}", task_type), 1).await?;
        
        // Update LLM metrics
        let mut profile = self.profile.write().await;
        
        // Update success rate
        let success_key = format!("{}_success_rate", llm_name);
        let usage_count = *profile.stats.get(&format!("llm_usage_{}", llm_name)).unwrap_or(&1);
        let success_count = *profile.stats.get(&format!("llm_success_{}", llm_name)).unwrap_or(&0);
        
        profile.llm_metrics.insert(
            success_key,
            success_count as f64 / usage_count as f64,
        );
        
        // Update average duration
        let duration_key = format!("{}_avg_duration", llm_name);
        let current_avg = profile.llm_metrics.get(&duration_key).copied().unwrap_or(0.0);
        
        profile.llm_metrics.insert(
            duration_key,
            if usage_count == 1 {
                duration_ms as f64
            } else {
                // Exponential moving average with alpha=0.1
                current_avg * 0.9 + (duration_ms as f64) * 0.1
            },
        );
        
        drop(profile);
        
        // Award XP for successful generations
        if success {
            self.award_xp(10, &format!("Successful {} generation with {}", task_type, llm_name)).await?;
        }
        
        self.save_profile().await?;
        
        Ok(())
    }
    
    /// Record code quality metrics
    pub async fn record_code_quality(&self, language: &str, metrics: HashMap<String, f64>) -> Result<(), AppError> {
        if !self.enabled {
            return Ok(());
        }
        
        let mut profile = self.profile.write().await;
        
        // Store metrics by language
        for (metric, value) in metrics {
            let key = format!("{}_{}", language, metric);
            profile.code_quality.insert(key, value);
        }
        
        drop(profile);
        self.save_profile().await?;
        
        Ok(())
    }
    
    /// Update daily streak
    pub async fn update_streak(&self) -> Result<(), AppError> {
        if !self.enabled {
            return Ok(());
        }
        
        let today = chrono::Local::now().date_naive().to_string();
        let mut _streak_days = 0; // Store the streak days value outside
        
        {
            let mut profile = self.profile.write().await;
            
            if profile.streak.last_date != today {
                let last_date = chrono::NaiveDate::parse_from_str(&profile.streak.last_date, "%Y-%m-%d")
                    .unwrap_or_else(|_| chrono::Local::now().date_naive());
                
                let today_date = chrono::NaiveDate::parse_from_str(&today, "%Y-%m-%d")
                    .unwrap_or_else(|_| chrono::Local::now().date_naive());
                
                // Calculate days between
                let days_between = (today_date - last_date).num_days();
                
                if days_between == 1 {
                    // Consecutive day
                    profile.streak.days += 1;
                    profile.streak.last_date = today;
                    
                    // Update longest streak if needed
                    if profile.streak.days > profile.streak.longest {
                        profile.streak.longest = profile.streak.days;
                    }
                    
                    // Save streak days value for use outside this scope
                    _streak_days = profile.streak.days;
                    
                    // Award streak XP (increasing with streak length)
                    let streak_xp = 10 + (profile.streak.days as u64 * 5).min(50);
                    drop(profile);
                    
                    self.award_xp(streak_xp, &format!("Daily streak: {} days", _streak_days)).await?;
                    
                    // Check for streak achievements
                    if _streak_days >= 7 {
                        self.award_achievement(AchievementType::FirstSuccessfulGeneration, "Week Warrior", 
                            "Maintain a 7-day streak", 100).await?;
                    }
                    
                    if _streak_days >= 30 {
                        self.award_achievement(AchievementType::FirstSuccessfulGeneration, "Month Master", 
                            "Maintain a 30-day streak", 500).await?;
                    }
                } else if days_between > 1 {
                    // Streak broken
                    profile.streak.days = 1;
                    profile.streak.last_date = today;
                }
            }
        }
        
        self.save_profile().await?;
        
        Ok(())
    }
    
    /// Award an achievement to the user
    pub async fn award_achievement(
        &self,
        achievement_type: AchievementType,
        name: &str,
        description: &str,
        xp_reward: u64,
    ) -> Result<bool, AppError> {
        if !self.enabled {
            return Ok(false);
        }
        
        // Check if already earned
        let mut profile = self.profile.write().await;
        let already_earned = profile.achievements.iter().any(|a| a.achievement_type == achievement_type);
        
        if already_earned {
            return Ok(false);
        }
        
        // Add achievement
        profile.achievements.push(Achievement {
            achievement_type,
            unlocked_at: None,
            progress: 0.0,
            description: description.to_string(),
        });
        
        drop(profile);
        
        // Award XP
        self.award_xp(xp_reward, &format!("Achievement: {}", name)).await?;
        
        // Save profile
        self.save_profile().await?;
        
        info!("Achievement unlocked: {}", name);
        Ok(true)
    }
    
    /// Check for achievements based on stats
    async fn check_stat_achievements(&self, stat_name: &str) -> Result<(), AppError> {
        let profile = self.profile.read().await;
        let stat_value = *profile.stats.get(stat_name).unwrap_or(&0);
        drop(profile);
        
        // PDF processing achievements
        if stat_name == "pdfs_processed" {
            if stat_value >= 1 {
                self.award_achievement(AchievementType::FirstSuccessfulGeneration, "First Steps", 
                    "Process your first PDF research paper", 50).await?;
            }
            
            if stat_value >= 10 {
                self.award_achievement(AchievementType::FirstSuccessfulGeneration, "Paper Reader", 
                    "Process 10 PDF research papers", 100).await?;
            }
            
            if stat_value >= 50 {
                self.award_achievement(AchievementType::FirstSuccessfulGeneration, "Research Enthusiast", 
                    "Process 50 PDF research papers", 300).await?;
            }
            
            if stat_value >= 100 {
                self.award_achievement(AchievementType::FirstSuccessfulGeneration, "Academic Explorer", 
                    "Process 100 PDF research papers", 500).await?;
            }
        }
        
        // Code generation achievements
        if stat_name == "code_generated" {
            if stat_value >= 1 {
                self.award_achievement(AchievementType::FirstCodeGeneration, "Code Apprentice", 
                    "Generate your first code snippet", 50).await?;
            }
            
            if stat_value >= 25 {
                self.award_achievement(AchievementType::FirstSuccessfulGeneration, "Code Craftsman", 
                    "Generate 25 code snippets", 150).await?;
            }
            
            if stat_value >= 100 {
                self.award_achievement(AchievementType::FirstSuccessfulGeneration, "Code Master", 
                    "Generate 100 code snippets", 400).await?;
            }
        }
        
        // Language diversity achievements
        if stat_name.starts_with("language_") {
            let profile = self.profile.read().await;
            let language_count = profile.stats.keys()
                .filter(|k| k.starts_with("language_"))
                .count();
            drop(profile);
            
            if language_count >= 3 {
                self.award_achievement(AchievementType::FirstSuccessfulGeneration, "Polyglot Programmer", 
                    "Generate code in 3 different programming languages", 100).await?;
            }
            
            if language_count >= 7 {
                self.award_achievement(AchievementType::FirstSuccessfulGeneration, "Language Virtuoso", 
                    "Generate code in 7 different programming languages", 300).await?;
            }
        }
        
        // LLM strategy achievements
        if stat_name.starts_with("llm_strategy_") {
            let strategy_name = stat_name.strip_prefix("llm_strategy_").unwrap_or("");
            
            self.award_achievement(AchievementType::FirstSuccessfulGeneration, &format!("Strategy Explorer: {}", strategy_name.replace('_', " ")),
                &format!("Use the {} LLM strategy", strategy_name.replace('_', " ")), 50).await?;
            
            // Check for strategy master
            let profile = self.profile.read().await;
            let strategy_count = profile.stats.keys()
                .filter(|k| k.starts_with("llm_strategy_"))
                .count();
            drop(profile);
            
            if strategy_count >= 5 {
                self.award_achievement(AchievementType::FirstSuccessfulGeneration, "Strategy Master", 
                    "Use all available LLM strategies", 200).await?;
            }
        }
        
        // Code quality achievements
        if stat_name == "high_quality_code" && stat_value >= 10 {
            self.award_achievement(AchievementType::FirstSuccessfulGeneration, "Quality Craftsman", 
                "Generate 10 high-quality code snippets", 200).await?;
        }
        
        Ok(())
    }
    
    /// Get user summary
    pub async fn get_user_summary(&self) -> Result<UserSummary, AppError> {
        let profile = self.profile.read().await;
        
        let summary = UserSummary {
            level: profile.level,
            xp: profile.xp,
            next_level_xp: profile.next_level_xp,
            achievements_count: profile.achievements.len(),
            recent_achievements: profile.achievements.iter()
                .rev()
                .take(3)
                .map(|a| (a.description.clone(), a.unlocked_at.map(|t| t.to_rfc3339()).unwrap_or_default()))
                .collect(),
            streak_days: profile.streak.days,
            longest_streak: profile.streak.longest,
            total_pdfs: *profile.stats.get("pdfs_processed").unwrap_or(&0),
            total_code: *profile.stats.get("code_generated").unwrap_or(&0),
            challenges: profile.challenges.iter()
                .filter(|c| !c.completed)
                .map(|c| (
                    c.name.clone(), 
                    c.progress as f64 / c.target as f64 * 100.0
                ))
                .collect(),
        };
        
        Ok(summary)
    }
    
    /// Get user challenges
    pub async fn get_user_challenges(&self) -> Result<Vec<Challenge>, AppError> {
        let profile = self.profile.read().await;
        let challenges = profile.challenges.clone();
        Ok(challenges)
    }
    
    /// Get user achievements
    pub async fn get_user_achievements(&self) -> Result<Vec<Achievement>, AppError> {
        let profile = self.profile.read().await;
        let achievements = profile.achievements.clone();
        Ok(achievements)
    }
}

/// User summary for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSummary {
    /// Current level
    pub level: u32,
    
    /// Current XP
    pub xp: u64,
    
    /// XP needed for next level
    pub next_level_xp: u64,
    
    /// Number of achievements earned
    pub achievements_count: usize,
    
    /// Recent achievements (name, date)
    pub recent_achievements: Vec<(String, String)>,
    
    /// Current streak
    pub streak_days: u32,
    
    /// Longest streak
    pub longest_streak: u32,
    
    /// Total PDFs processed
    pub total_pdfs: u64,
    
    /// Total code snippets generated
    pub total_code: u64,
    
    /// Active challenges (name, percent complete)
    pub challenges: Vec<(String, f64)>,
}

/// Code quality analyzer
#[derive(Debug, Clone)]
pub struct CodeQualityAnalyzer {
    /// Gamification system
    gamification: Option<Arc<GamificationSystem>>,
}

impl CodeQualityAnalyzer {
    /// Create a new code quality analyzer
    pub fn new(gamification: Option<Arc<GamificationSystem>>) -> Self {
        Self { gamification }
    }
    
    /// Analyze code quality and award achievements/XP
    pub async fn analyze_code_quality(
        &self,
        code: &str,
        language: &str,
        _task_type: TaskType,
    ) -> Result<HashMap<String, f64>, AppError> {
        // Calculate various code quality metrics
        let metrics = self.calculate_metrics(code, language);
        
        // Get overall score
        let overall_score = self.calculate_overall_score(&metrics, language);
        
        // Update gamification if enabled
        if let Some(gamification) = &self.gamification {
            // Record quality metrics
            let mut gamification_metrics = metrics.clone();
            gamification_metrics.insert("overall_score".to_string(), overall_score);
            gamification.record_code_quality(language, gamification_metrics).await?;
            
            // Award XP based on quality
            let quality_xp = if overall_score >= 90.0 {
                gamification.update_stat("high_quality_code", 1).await?;
                50
            } else if overall_score >= 75.0 {
                25
            } else if overall_score >= 60.0 {
                10
            } else {
                5
            };
            
            gamification.award_xp(
                quality_xp,
                &format!("Code quality score: {:.1} for {} code", overall_score, language),
            ).await?;
        }
        
        // Return metrics including overall score
        let mut result = metrics;
        result.insert("overall_score".to_string(), overall_score);
        Ok(result)
    }
    
    /// Calculate code quality metrics
    fn calculate_metrics(&self, code: &str, language: &str) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        // Basic metrics
        let lines = code.lines().count();
        metrics.insert("lines".to_string(), lines as f64);
        
        let chars = code.chars().count();
        metrics.insert("characters".to_string(), chars as f64);
        
        // Calculate non-empty lines
        let non_empty_lines = code.lines()
            .filter(|line| !line.trim().is_empty())
            .count();
        metrics.insert("non_empty_lines".to_string(), non_empty_lines as f64);
        
        // Calculate comment lines
        let comment_prefix = match language {
            "python" | "ruby" => "#",
            "javascript" | "typescript" | "java" | "c" | "cpp" | "rust" => "//",
            "lua" => "--",
            "lisp" | "clojure" => ";",
            _ => "#",
        };
        
        let comment_lines = code.lines()
            .filter(|line| line.trim().starts_with(comment_prefix))
            .count();
        metrics.insert("comment_lines".to_string(), comment_lines as f64);
        
        // Calculate comment ratio
        if non_empty_lines > 0 {
            let comment_ratio = comment_lines as f64 / non_empty_lines as f64;
            metrics.insert("comment_ratio".to_string(), comment_ratio);
        }
        
        // Calculate average line length
        if lines > 0 {
            let avg_line_length = chars as f64 / lines as f64;
            metrics.insert("avg_line_length".to_string(), avg_line_length);
        }
        
        // Calculate function count (simple heuristic)
        let function_keyword = match language {
            "python" => "def ",
            "javascript" | "typescript" => "function ",
            "java" | "c" | "cpp" => ") {",
            "rust" => "fn ",
            "ruby" => "def ",
            "go" => "func ",
            _ => "def ",
        };
        
        let function_count = code.lines()
            .filter(|line| line.contains(function_keyword))
            .count();
        metrics.insert("function_count".to_string(), function_count as f64);
        
        // Complexity metrics (simplified)
        let is_complex_lang = matches!(
            language,
            "python" | "javascript" | "typescript" | "java" | "c" | "cpp" | "rust" | "go"
        );
        
        // Count control flow statements as a proxy for cyclomatic complexity
        if is_complex_lang {
            let control_flow_keywords = [
                "if ", "else ", "for ", "while ", "switch ", "case ", "match ", "catch ", "try "
            ];
            
            let control_flow_count = code.lines()
                .filter(|line| {
                    control_flow_keywords.iter().any(|kw| line.contains(kw))
                })
                .count();
            
            metrics.insert("control_flow_count".to_string(), control_flow_count as f64);
            
            if function_count > 0 {
                let complexity_per_function = control_flow_count as f64 / function_count as f64;
                metrics.insert("complexity_per_function".to_string(), complexity_per_function);
            }
        }
        
        metrics
    }
    
    /// Calculate overall quality score (0-100)
    fn calculate_overall_score(&self, metrics: &HashMap<String, f64>, language: &str) -> f64 {
        let mut score = 75.0; // Start with a baseline score
        
        // Comment ratio (ideal: 0.15-0.3)
        if let Some(comment_ratio) = metrics.get("comment_ratio") {
            if *comment_ratio < 0.1 {
                score -= 10.0 * (0.1 - comment_ratio); // Penalize under-commented code
            } else if *comment_ratio > 0.5 {
                score -= 5.0 * (comment_ratio - 0.5); // Minor penalty for over-commenting
            } else if (0.15..=0.3).contains(comment_ratio) {
                score += 5.0; // Bonus for ideal comment ratio
            }
        }
        
        // Average line length (ideal: 30-80 chars)
        if let Some(avg_line_length) = metrics.get("avg_line_length") {
            if *avg_line_length > 100.0 {
                score -= 10.0 * ((*avg_line_length - 100.0) / 50.0).min(1.0); // Penalize very long lines
            } else if (30.0..=80.0).contains(avg_line_length) {
                score += 5.0; // Bonus for ideal line length
            }
        }
        
        // Function count (encourage modular code)
        if let Some(function_count) = metrics.get("function_count") {
            let non_empty_lines = metrics.get("non_empty_lines").copied().unwrap_or(1.0);
            let lines_per_function = if *function_count > 0.0 {
                non_empty_lines / function_count
            } else {
                non_empty_lines
            };
            
            // Ideal: 5-20 lines per function
            if lines_per_function > 30.0 {
                score -= 10.0 * ((lines_per_function - 30.0) / 20.0).min(1.0); // Penalize long functions
            } else if (5.0..=20.0).contains(&lines_per_function) {
                score += 5.0; // Bonus for ideal function size
            }
        }
        
        // Complexity per function (lower is better)
        if let Some(complexity_per_function) = metrics.get("complexity_per_function") {
            if *complexity_per_function > 5.0 {
                score -= 10.0 * ((*complexity_per_function - 5.0) / 5.0).min(1.0); // Penalize complex functions
            } else if *complexity_per_function <= 3.0 {
                score += 5.0; // Bonus for simple functions
            }
        }
        
        // Adjust based on language-specific expectations
        match language {
            "rust" => {
                // Rust code often has fewer comments but more descriptive names
                if let Some(comment_ratio) = metrics.get("comment_ratio") {
                    if *comment_ratio >= 0.1 {
                        score += 5.0; // Less strict on Rust comments
                    }
                }
            },
            "python" => {
                // Python values readability
                if let Some(avg_line_length) = metrics.get("avg_line_length") {
                    if *avg_line_length <= 79.0 {
                        score += 5.0; // PEP 8 line length
                    }
                }
            },
            _ => {}
        }
        
        // Ensure score is within 0-100 range
        score.max(0.0).min(100.0)
    }
}

/// Gamification for CLI display
pub struct GamificationDisplay {
    system: Arc<GamificationSystem>,
}

impl GamificationDisplay {
    /// Create a new gamification display
    pub fn new(system: Arc<GamificationSystem>) -> Self {
        Self { system }
    }
    
    /// Display user's current status
    pub async fn display_status(&self) -> Result<String, AppError> {
        let summary = self.system.get_user_summary().await?;
        
        let mut output = String::new();
        output.push_str(&format!("🏆 Level: {} | XP: {}/{}\n", 
            summary.level, summary.xp, summary.next_level_xp));
        
        output.push_str(&format!("🔥 Streak: {} days (Record: {})\n", 
            summary.streak_days, summary.longest_streak));
        
        output.push_str(&format!("🎓 Achievements: {}\n", 
            summary.achievements_count));
        
        if !summary.recent_achievements.is_empty() {
            output.push_str("🌟 Recent: ");
            for (i, (name, _)) in summary.recent_achievements.iter().enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                output.push_str(name);
            }
            output.push('\n');
        }
        
        if !summary.challenges.is_empty() {
            output.push_str("📋 Active challenges:\n");
            for (name, percent) in &summary.challenges {
                output.push_str(&format!("   ▪ {} - {:.1}%\n", name, percent));
            }
        }
        
        Ok(output)
    }
    
    /// Format achievement notification
    pub fn format_achievement_notification(name: &str, description: &str, xp: u64) -> String {
        format!("🏆 Achievement unlocked: {} 🏆\n{}\n+{} XP", name, description, xp)
    }
    
    /// Format level up notification
    pub fn format_level_up_notification(level: u32) -> String {
        format!("🌟 LEVEL UP! 🌟\nYou are now level {}!", level)
    }
}