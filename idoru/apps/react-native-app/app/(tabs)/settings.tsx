import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Alert,
  RefreshControl,
  SafeAreaView,
  Platform,
} from "react-native";
import {
  Ionicons,
  FontAwesome,
  FontAwesome5,
  FontAwesome6,
} from "@expo/vector-icons";
import * as WebBrowser from "expo-web-browser";
import { useRouter } from "expo-router";
import { socialApiService } from "../../services/socialApi";
import { useAuthFlow } from "../../hooks/useAuthFlow";
import { StatusBar } from "expo-status-bar";

interface SocialPlatform {
  id: string;
  name: string;
  iconLibrary: "FontAwesome" | "FontAwesome5" | "Ionicons";
  iconName: string;
  connected: boolean;
  color: string;
  isAvailable: boolean;
}

const SettingsScreen: React.FC = () => {
  const router = useRouter();
  const { state } = useAuthFlow();
  const [platforms, setPlatforms] = useState<SocialPlatform[]>([
    {
      id: "instagram",
      name: "Instagram",
      iconLibrary: "FontAwesome",
      iconName: "instagram",
      connected: false,
      color: "#E4405F",
      isAvailable: true,
    },
    {
      id: "tiktok",
      name: "TikTok",
      iconLibrary: "FontAwesome5",
      iconName: "tiktok",
      connected: false,
      color: "#fff",
      isAvailable: true,
    },
    {
      id: "twitter",
      name: "X (Twitter)",
      iconLibrary: "FontAwesome6",
      iconName: "x",
      connected: false,
      color: "#fff",
      isAvailable: true,
    },
    {
      id: "bluesky",
      name: "Bluesky",
      iconLibrary: "Ionicons",
      iconName: "cloud-outline",
      connected: false,
      color: "#00A8E8",
      isAvailable: false,
    },
    {
      id: "facebook",
      name: "Facebook",
      iconLibrary: "FontAwesome",
      iconName: "facebook",
      connected: false,
      color: "#4267B2",
      isAvailable: false,
    },

    {
      id: "linkedin",
      name: "LinkedIn",
      iconLibrary: "FontAwesome",
      iconName: "linkedin",
      connected: false,
      color: "#0077B5",
      isAvailable: false,
    },
    {
      id: "pinterest",
      name: "Pinterest",
      iconLibrary: "FontAwesome",
      iconName: "pinterest",
      connected: false,
      color: "#BD081C",
      isAvailable: false,
    },
    {
      id: "reddit",
      name: "Reddit",
      iconLibrary: "FontAwesome",
      iconName: "reddit",
      connected: false,
      color: "#FF4500",
      isAvailable: false,
    },
    {
      id: "snapchat",
      name: "Snapchat",
      iconLibrary: "FontAwesome",
      iconName: "snapchat",
      connected: false,
      color: "#FFFC00",
      isAvailable: false,
    },
    {
      id: "telegram",
      name: "Telegram",
      iconLibrary: "FontAwesome",
      iconName: "telegram",
      connected: false,
      color: "#0088CC",
      isAvailable: false,
    },
    {
      id: "threads",
      name: "Threads",
      iconLibrary: "Ionicons",
      iconName: "git-branch-outline",
      connected: false,
      color: "#000000",
      isAvailable: false,
    },

    // {
    //     id: 'youtube',
    //     name: 'YouTube',
    //     iconLibrary: 'FontAwesome',
    //     iconName: 'youtube',
    //     connected: false,
    //     isAvailable: false,
    //     color: '#FF0000',
    // },
  ]);

  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(false);

  // Fetch integration status from your API
  const fetchIntegrationStatus = async () => {
    try {
      console.log("Fetching integration status...");

      const data = await socialApiService.getConnectionStatus();

      if (data.success && data.platforms) {
        // Update platforms with real status from backend
        setPlatforms((prevPlatforms) =>
          prevPlatforms.map((platform) => ({
            ...platform,
            connected: data.platforms[platform.id]?.connected || false,
          }))
        );
      }
    } catch (error) {
      console.error("Error fetching integration status:", error);
      // Alert.alert(
      //     'Connection Error',
      //     'Failed to fetch social media status. Please check your connection and try again.'
      // );
    }
  };

  // Handle authenticated navigation to web management
  const handleManageIntegrations = async () => {
    try {
      setLoading(true);

      const data = await socialApiService.generateConnectionUrls();
      if (data?.url) {
        // Open the connection URL in browser
        const result = await WebBrowser.openBrowserAsync(data.url, {
          presentationStyle: WebBrowser.WebBrowserPresentationStyle.FORM_SHEET,
          controlsColor: "#007AFF",
        });

        // When browser closes, refresh the integration status
        if (result.type === "dismiss") {
          await fetchIntegrationStatus();
        }
      } else {
        Alert.alert("Error", "Failed to get connection URL from server");
      }
    } catch (error) {
      console.error("Error generating connection URLs:", error);
      Alert.alert(
        "Connection Error",
        "Failed to generate social media connection URLs. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchIntegrationStatus();
    setRefreshing(false);
  };

  useEffect(() => {
    // Fetch status on mount and when auth state changes
    fetchIntegrationStatus();

    // Set up polling interval (every 30 seconds)
    const interval = setInterval(fetchIntegrationStatus, 30000);

    return () => clearInterval(interval);
  }, [state]);

  const renderIcon = (platform: SocialPlatform) => {
    const iconSize = 24;
    const iconColor = platform.color;

    switch (platform.iconLibrary) {
      case "FontAwesome":
        return (
          <FontAwesome
            name={platform.iconName as any}
            size={iconSize}
            color={iconColor}
          />
        );
      case "FontAwesome5":
        return (
          <FontAwesome5
            name={platform.iconName as any}
            size={iconSize}
            color={iconColor}
          />
        );
      case "Ionicons":
        return (
          <Ionicons
            name={platform.iconName as any}
            size={iconSize}
            color={iconColor}
          />
        );
      case "FontAwesome6":
        return (
          <FontAwesome6
            name={platform.iconName as any}
            size={iconSize}
            color={iconColor}
          />
        );
      default:
        return (
          <Ionicons
            name="share-social-outline"
            size={iconSize}
            color={iconColor}
          />
        );
    }
  };

  const renderPlatformItem = ({ item }: { item: SocialPlatform }) => (
    <View
      style={[
        styles.platformItem,
        {
          opacity: item.isAvailable ? 1 : 0.6,
        },
      ]}
    >
      <View style={styles.platformInfo}>
        <View
          style={[styles.logoContainer, { backgroundColor: item.color + "20" }]}
        >
          {renderIcon(item)}
        </View>

        <View style={styles.platformDetails}>
          <Text style={styles.platformName}>{item.name}</Text>
          {!item.isAvailable && (
            <View
              style={{
                backgroundColor: "#666",
                paddingHorizontal: 4,
                paddingVertical: 2,
                borderRadius: 4,
              }}
            >
              <Text
                style={[
                  styles.platformSubtitle,
                  {
                    color: "#fff",
                  },
                ]}
              >
                Soon
              </Text>
            </View>
          )}
        </View>
      </View>

      {item.isAvailable && (
        <View style={styles.statusContainer}>
          <Text
            style={[
              styles.statusText,
              { color: item.connected ? "#4CAF50" : "#FF5722" },
            ]}
          >
            {item.connected ? "Connected" : "Not Connected"}
          </Text>
        </View>
      )}
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      {/* Header with Manage Button */}
      <StatusBar style="light" />
      <View style={styles.header}>
        <View style={styles.headerSpacer} />

        <TouchableOpacity
          style={[styles.manageButton, loading && styles.manageButtonDisabled]}
          onPress={handleManageIntegrations}
          disabled={loading}
        >
          {loading ? (
            <Ionicons name="hourglass-outline" size={20} color="#666" />
          ) : (
            <Ionicons name="settings-outline" size={20} color="#fff" />
          )}
          <Text
            style={[
              styles.manageButtonText,
              loading && styles.manageButtonTextDisabled,
            ]}
          >
            {loading ? "Connecting..." : "Manage"}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Section Title */}
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>Social Media Platforms</Text>
        <Text style={styles.sectionSubtitle}>
          Connect your social media accounts to share your content across
          platforms
        </Text>
      </View>

      {/* Platforms List */}
      <FlatList
        data={platforms}
        renderItem={renderPlatformItem}
        keyExtractor={(item) => item.name}
        contentContainerStyle={{
          paddingBottom: 20,
        }}
        style={styles.list}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        showsVerticalScrollIndicator={false}
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#111111",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingVertical: 8,
    backgroundColor: "#111111",
    borderBottomWidth: 1,
    borderBottomColor: "#282828",
    marginTop: Platform.OS === "ios" ? 0 : 20,
  },
  headerSpacer: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: "600",
    color: "white",
  },
  manageButton: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: "#282828",
    borderRadius: 16,
  },
  manageButtonDisabled: {
    backgroundColor: "#1a1a1a",
  },
  manageButtonText: {
    fontSize: 14,
    color: "#fff",
    marginLeft: 4,
    fontWeight: "500",
  },
  manageButtonTextDisabled: {
    color: "#666",
  },
  sectionHeader: {
    paddingHorizontal: 20,
    paddingVertical: 14,
    backgroundColor: "#111111",
    borderBottomWidth: 1,
    borderBottomColor: "#282828",
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "white",
    marginBottom: 4,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: "#808080",
    lineHeight: 20,
  },
  list: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 16,
  },
  platformItem: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: "#1a1a1a",
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#282828",
  },
  platformInfo: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
  },
  logoContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },
  platformDetails: {
    flexDirection: "row",
    flex: 1,
    gap: 4,
  },
  platformName: {
    fontSize: 16,
    fontWeight: "600",
    color: "white",
  },
  platformSubtitle: {
    fontSize: 12,
    color: "#666",
    fontWeight: "400",
  },
  statusContainer: {
    alignItems: "flex-end",
    justifyContent: "center",
  },
  statusText: {
    fontSize: 14,
    fontWeight: "500",
  },
});

export default SettingsScreen;
